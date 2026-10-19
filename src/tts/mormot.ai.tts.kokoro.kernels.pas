/// Text-to-speech - Kokoro's compute kernels on the CPU (x86_64 AVX2 / FMA, detected at runtime; Pascal loops otherwise)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the matrix product: a register-blocked micro-kernel, 6 rows x 16
// outputs (12 ymm accumulators), over weights packed in panels of 16
// outputs
// - y[r][o] := bias[o] + the sum over the taps k and the inputs c of
// x[r + k * dil][c] * w[o][c][k]: a stride-1 1D convolution read in place
// (im2col done on the fly), or a plain product (a single tap)
// - the sum runs in blocks of KGEMM_KB inputs of one tap: each block starts
// from zero in the registers (single) and is added to an intermediate total
// (single); every KGEMM_FLUSH blocks and at the end of each tap, the
// intermediate total is added to the total, in double, which starts from
// the bias; the total is rounded to single at the end - a 2816-term sum done
// in one go in single drifted four times further from PyTorch; this way
// the error stays the one of an all-double product
// - each output sums its terms in the same order whatever the tiling or the
// number of threads
// - AdaIN and Snake of the vocoder: the per-channel sums (in double) of the
// mean and the variance, then the normalization and Snake in one pass; the
// same operations, in the same order, as the Pascal loop: the same bits
// - the assembler body takes a single record, its parameters then room for
// the registers to restore: the same for Win64 and System V, as the kernels
// of mormot2-ai (mormot.ai.infer.kernels)
unit mormot.ai.tts.kokoro.kernels;

{$I mormot.defines.inc}

{$ifndef FPC}
  {$EXCESSPRECISION OFF} // single in single, as PyTorch's float32
{$endif FPC}

interface

uses
  sysutils,
  mormot.core.base;

const
  /// the outputs of a panel
  KGEMM_NR = 16;
  /// the rows of the micro-kernel
  KGEMM_MR = 6;
  /// flag: y := y + product (the residual), otherwise y := product
  KGEMM_ADD = 1;

type
  /// a matrix laid out for the kernel: for each panel of 16 outputs,
  // Ktot = Taps x Cin rows of 16 weights, tap by tap; the outputs beyond
  // OutCount (the last panel) have zero weights and bias
  TKPanels = record
    W: TSingleDynArray;
    B: TSingleDynArray;
    Cin, Taps, Ktot, OutCount, Panels: PtrInt;
  end;
  PKPanels = ^TKPanels;

/// lays out w[aOut][aCin][aTaps] (PyTorch's order; aTaps = 1 for a plain
// product) and the bias b[aOut] (nil: zero) in m
procedure KokoroPack(var m: TKPanels; w: PSingleArray; aOut, aCin, aTaps: PtrInt;
  b: PSingleArray);

/// aRows rows of the panel aPanel: x = row 0 at tap 0, aXStride bytes
// from one row to the next, aTapStride bytes from one tap to the next; y =
// row 0 at column aPanel x 16, aYStride bytes per row
procedure KokoroGemmPanel(x: pointer; aXStride, aTapStride: PtrInt;
  const m: TKPanels; aPanel: PtrInt; y: pointer; aYStride, aRows: PtrInt;
  aFlags: PtrInt);

/// the panels aPanel and aPanel + 1, complete, together: as two
// KokoroGemmPanel, faster with AVX-512
procedure KokoroGemmPair(x: pointer; aXStride, aTapStride: PtrInt;
  const m: TKPanels; aPanel: PtrInt; y: pointer; aYStride, aRows: PtrInt;
  aFlags: PtrInt);

/// true: KokoroGemmPair has a kernel of its own (AVX-512)
function KokoroGemmPairs: boolean;

/// 'avx512', 'avx2' or 'pascal'
function KokoroGemmKernel: RawUtf8;

/// sin(x) in single, within a few ulp for |x| < 10^4: Cody-Waite
// reduction by pi / 2, then the Cephes polynomials (sinf, cosf)
function KokoroSin(x: single): single;

/// aPart[c] := aPart[c] + the sum of x[r * aStride + c] (in double), for
// r < aRows and c < aCols
procedure KokoroColSums(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  aPart: PDoubleArray);

/// aPart[c] := aPart[c] + the sum of (x[r * aStride + c] - aMean[c])^2
procedure KokoroColSquares(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  aMean, aPart: PDoubleArray);

/// y[r][c] := v + aInv[c] * sin(aAlpha[c] * v)^2 (Snake), with
// v := x[r][c] * aScale[c] + aShift[c] (AdaIN); aScale = nil: v := x[r][c];
// aAlpha = nil: y[r][c] := v; y may be x; the strides in singles
procedure KokoroAffineSnake(x: PSingleArray; aRows, aCols, aXStride: PtrInt;
  y: PSingleArray; aYStride: PtrInt; aScale, aShift, aAlpha, aInv: PSingleArray);

/// one step of an LSTM (PyTorch's gates i, f, g, o): gx the inputs
// x W_ih^T + b_ih, gh the h W_hh^T, b the b_hh, [4][aHidden] each; c and
// h [aHidden] updated in place
procedure KokoroLstmCell(gx, gh, b, c, h: PSingleArray; aHidden: PtrInt);

/// tests: true = the Pascal loops instead of the AVX2 kernels
procedure KokoroForcePascal(aPascal: boolean);

const
  /// the inputs of a block of the product sum
  KGEMM_KB = 16;
  /// the blocks of an intermediate total, at most
  KGEMM_FLUSH = 4;


implementation

type
  /// the single argument of the kernels: the parameters, room for the
  // registers the ABI wants back, then the totals of the 6 rows
  TKGemmArgs = packed record
    /// x, xs, w, cin, taps, ts, y, ys, bias, flags, kb, panel gap:
    // offsets 0 .. 88
    A: array[0..11] of PtrInt;
    /// rbx rsi rdi r12 r13 r14: offsets 96 .. 136
    G: array[0..7] of PtrInt;
    /// xmm6 .. xmm15 (Win64 only): offsets 160 .. 304
    X: array[0..9] of THash128;
    /// the totals in double, row i at offsets 320 + 256 i (16 or 32
    // outputs)
    T: array[0..KGEMM_MR * 2 * KGEMM_NR - 1] of double;
    /// the intermediate totals in single (AVX2), row i at offsets
    // 1856 + 128 i
    M: array[0..KGEMM_MR * 2 * KGEMM_NR - 1] of single;
  end;

  TKGemmKernel = procedure(var aArgs: TKGemmArgs);

const
  /// the constants of KokoroSin: 2 / pi, 1 / 2, pi / 2 in three pieces,
  // the polynomials of sinf (S1..S3) and cosf (C1..C3), 1
  SNAKE_CONSTS: array[0..11] of single = (
    0.63661977236758134, 0.5, 1.5703125, 4.837512969970703125e-4,
    7.549789948768648e-8, -1.6666654611e-1, 8.3321608736e-3, -1.9515295891e-4,
    4.166664568298827e-2, -1.388731625493765e-3, 2.443315711809948e-5, 1);

var
  /// 6 rows x 16 outputs, and 1 row x 16 outputs
  KGemm6: TKGemmKernel;
  KGemm1: TKGemmKernel;
  /// 6 rows x 32 outputs; nil without AVX-512
  KGemm6x32: TKGemmKernel;
  KGemmName: RawUtf8;
  KHasAvx2: boolean;
  /// SNAKE_CONSTS on 8 lanes, then |x|, the sign, 1 and 2 as integers
  KSnakeConsts: array[0..15] of array[0..7] of single;
  /// 1 / ln 2, ln 2 in two pieces, 1 / k! (k = 0 .. 13), 1, the sign,
  // 1023, the bounds +-700 and +-20, each on 4 lanes
  KLstmConsts: array[0..23] of array[0..3] of double;

/// the Pascal loop, aRows rows of 16 outputs (no FMA), by the same
// blocks, intermediate totals and totals as the kernels
procedure KGemmPascal(var aArgs: TKGemmArgs; aRows: PtrInt);
var
  r, k, c, j, left, n, blocks: PtrInt;
  acc, mid: array[0..KGEMM_NR - 1] of single;
  total: array[0..KGEMM_NR - 1] of double;
  xr, wr, yr, bias: PSingleArray;
  v: single;
begin
  bias := pointer(aArgs.A[8]);
  for r := 0 to aRows - 1 do
  begin
    for j := 0 to KGEMM_NR - 1 do
      total[j] := bias^[j];
    wr := pointer(aArgs.A[2]);
    for k := 0 to aArgs.A[4] - 1 do
    begin
      xr := pointer(aArgs.A[0] + r * aArgs.A[1] + k * aArgs.A[5]);
      left := aArgs.A[3];
      FillCharFast(mid, SizeOf(mid), 0);
      blocks := KGEMM_FLUSH;
      while left > 0 do
      begin
        n := left;
        if n > aArgs.A[10] then
          n := aArgs.A[10];
        dec(left, n);
        FillCharFast(acc, SizeOf(acc), 0);
        for c := 0 to n - 1 do
        begin
          v := xr^[c];
          for j := 0 to KGEMM_NR - 1 do
            acc[j] := acc[j] + v * wr^[j];
          wr := @wr^[KGEMM_NR];
        end;
        xr := @xr^[n];
        for j := 0 to KGEMM_NR - 1 do
          mid[j] := acc[j] + mid[j];
        dec(blocks);
        if (blocks = 0) or (left = 0) then
        begin
          for j := 0 to KGEMM_NR - 1 do
            total[j] := mid[j] + total[j];
          FillCharFast(mid, SizeOf(mid), 0);
          blocks := KGEMM_FLUSH;
        end;
      end;
    end;
    yr := pointer(aArgs.A[6] + r * aArgs.A[7]);
    for j := 0 to KGEMM_NR - 1 do
    begin
      v := total[j];
      if aArgs.A[9] and KGEMM_ADD <> 0 then
        v := v + yr^[j];
      yr^[j] := v;
    end;
  end;
end;

procedure KGemm6Pascal(var aArgs: TKGemmArgs);
begin
  KGemmPascal(aArgs, KGEMM_MR);
end;

procedure KGemm1Pascal(var aArgs: TKGemmArgs);
begin
  KGemmPascal(aArgs, 1);
end;

{$ifdef CPUX64}
{$ifdef ASMX64AVX1}

{$define KGEMMAVX2}

/// 6 rows x 16 outputs: ymm0 .. ymm11 the accumulators (row i: ymm2i
// and ymm2i+1), ymm12 / ymm13 the 16 weights of the input, ymm14 / ymm15 the
// broadcast inputs; the intermediate totals in memory
// - rbx = the current tap (row 0), rsi = its row 0, r8 = its row 3,
// rdx = xs, r9 = the weights, rax = the inputs left in the tap, rcx = those
// of the block, rdi = the blocks before the move to double, r10 = the taps
// left
procedure KGemm6Avx2(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        mov     qword ptr [r11 + 96], rbx
        mov     qword ptr [r11 + 104], rsi
        mov     qword ptr [r11 + 112], rdi
        mov     qword ptr [r11 + 120], r12
        mov     qword ptr [r11 + 128], r13
        {$ifdef ABIWINX64}
        vmovups oword ptr [r11 + 160], xmm6
        vmovups oword ptr [r11 + 176], xmm7
        vmovups oword ptr [r11 + 192], xmm8
        vmovups oword ptr [r11 + 208], xmm9
        vmovups oword ptr [r11 + 224], xmm10
        vmovups oword ptr [r11 + 240], xmm11
        vmovups oword ptr [r11 + 256], xmm12
        vmovups oword ptr [r11 + 272], xmm13
        vmovups oword ptr [r11 + 288], xmm14
        vmovups oword ptr [r11 + 304], xmm15
        {$endif ABIWINX64}
        // the totals start from the bias, in double; the intermediate ones from zero
        mov     rax, qword ptr [r11 + 64]
        vcvtps2pd ymm0, oword ptr [rax]
        vcvtps2pd ymm1, oword ptr [rax + 16]
        vcvtps2pd ymm2, oword ptr [rax + 32]
        vcvtps2pd ymm3, oword ptr [rax + 48]
        vmovupd yword ptr [r11 + 320], ymm0
        vmovupd yword ptr [r11 + 352], ymm1
        vmovupd yword ptr [r11 + 384], ymm2
        vmovupd yword ptr [r11 + 416], ymm3
        vmovupd yword ptr [r11 + 576], ymm0
        vmovupd yword ptr [r11 + 608], ymm1
        vmovupd yword ptr [r11 + 640], ymm2
        vmovupd yword ptr [r11 + 672], ymm3
        vmovupd yword ptr [r11 + 832], ymm0
        vmovupd yword ptr [r11 + 864], ymm1
        vmovupd yword ptr [r11 + 896], ymm2
        vmovupd yword ptr [r11 + 928], ymm3
        vmovupd yword ptr [r11 + 1088], ymm0
        vmovupd yword ptr [r11 + 1120], ymm1
        vmovupd yword ptr [r11 + 1152], ymm2
        vmovupd yword ptr [r11 + 1184], ymm3
        vmovupd yword ptr [r11 + 1344], ymm0
        vmovupd yword ptr [r11 + 1376], ymm1
        vmovupd yword ptr [r11 + 1408], ymm2
        vmovupd yword ptr [r11 + 1440], ymm3
        vmovupd yword ptr [r11 + 1600], ymm0
        vmovupd yword ptr [r11 + 1632], ymm1
        vmovupd yword ptr [r11 + 1664], ymm2
        vmovupd yword ptr [r11 + 1696], ymm3
        vxorps  ymm4, ymm4, ymm4
        vmovups yword ptr [r11 + 1856], ymm4
        vmovups yword ptr [r11 + 1888], ymm4
        vmovups yword ptr [r11 + 1984], ymm4
        vmovups yword ptr [r11 + 2016], ymm4
        vmovups yword ptr [r11 + 2112], ymm4
        vmovups yword ptr [r11 + 2144], ymm4
        vmovups yword ptr [r11 + 2240], ymm4
        vmovups yword ptr [r11 + 2272], ymm4
        vmovups yword ptr [r11 + 2368], ymm4
        vmovups yword ptr [r11 + 2400], ymm4
        vmovups yword ptr [r11 + 2496], ymm4
        vmovups yword ptr [r11 + 2528], ymm4
        mov     rbx, qword ptr [r11]
        mov     rdx, qword ptr [r11 + 8]
        mov     r9, qword ptr [r11 + 16]
        mov     r10, qword ptr [r11 + 32]
@tap:   mov     rsi, rbx
        lea     r8, [rbx + rdx * 2]
        add     r8, rdx
        mov     rax, qword ptr [r11 + 24]
        mov     edi, KGEMM_FLUSH
@block: mov     rcx, qword ptr [r11 + 80]
        cmp     rax, rcx
        cmovb   rcx, rax
        sub     rax, rcx
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        vxorps  ymm4, ymm4, ymm4
        vxorps  ymm5, ymm5, ymm5
        vxorps  ymm6, ymm6, ymm6
        vxorps  ymm7, ymm7, ymm7
        vxorps  ymm8, ymm8, ymm8
        vxorps  ymm9, ymm9, ymm9
        vxorps  ymm10, ymm10, ymm10
        vxorps  ymm11, ymm11, ymm11
@c:     vmovups ymm12, yword ptr [r9]
        vmovups ymm13, yword ptr [r9 + 32]
        vbroadcastss ymm14, dword ptr [rsi]
        vbroadcastss ymm15, dword ptr [rsi + rdx]
        vfmadd231ps ymm0, ymm14, ymm12
        vfmadd231ps ymm1, ymm14, ymm13
        vfmadd231ps ymm2, ymm15, ymm12
        vfmadd231ps ymm3, ymm15, ymm13
        vbroadcastss ymm14, dword ptr [rsi + rdx * 2]
        vbroadcastss ymm15, dword ptr [r8]
        vfmadd231ps ymm4, ymm14, ymm12
        vfmadd231ps ymm5, ymm14, ymm13
        vfmadd231ps ymm6, ymm15, ymm12
        vfmadd231ps ymm7, ymm15, ymm13
        vbroadcastss ymm14, dword ptr [r8 + rdx]
        vbroadcastss ymm15, dword ptr [r8 + rdx * 2]
        vfmadd231ps ymm8, ymm14, ymm12
        vfmadd231ps ymm9, ymm14, ymm13
        vfmadd231ps ymm10, ymm15, ymm12
        vfmadd231ps ymm11, ymm15, ymm13
        add     rsi, 4
        add     r8, 4
        add     r9, 64
        sub     rcx, 1
        jnz     @c
        // the block is added to the intermediate total
        vaddps  ymm0, ymm0, yword ptr [r11 + 1856]
        vaddps  ymm1, ymm1, yword ptr [r11 + 1888]
        vaddps  ymm2, ymm2, yword ptr [r11 + 1984]
        vaddps  ymm3, ymm3, yword ptr [r11 + 2016]
        vaddps  ymm4, ymm4, yword ptr [r11 + 2112]
        vaddps  ymm5, ymm5, yword ptr [r11 + 2144]
        vaddps  ymm6, ymm6, yword ptr [r11 + 2240]
        vaddps  ymm7, ymm7, yword ptr [r11 + 2272]
        vaddps  ymm8, ymm8, yword ptr [r11 + 2368]
        vaddps  ymm9, ymm9, yword ptr [r11 + 2400]
        vaddps  ymm10, ymm10, yword ptr [r11 + 2496]
        vaddps  ymm11, ymm11, yword ptr [r11 + 2528]
        sub     edi, 1
        jz      @flush
        test    rax, rax
        jz      @flush
        vmovups yword ptr [r11 + 1856], ymm0
        vmovups yword ptr [r11 + 1888], ymm1
        vmovups yword ptr [r11 + 1984], ymm2
        vmovups yword ptr [r11 + 2016], ymm3
        vmovups yword ptr [r11 + 2112], ymm4
        vmovups yword ptr [r11 + 2144], ymm5
        vmovups yword ptr [r11 + 2240], ymm6
        vmovups yword ptr [r11 + 2272], ymm7
        vmovups yword ptr [r11 + 2368], ymm8
        vmovups yword ptr [r11 + 2400], ymm9
        vmovups yword ptr [r11 + 2496], ymm10
        vmovups yword ptr [r11 + 2528], ymm11
        jmp     @next
        // every KGEMM_FLUSH blocks and at the end of a tap: into the total in
        // double, then the intermediate one back to zero
@flush:
        vcvtps2pd ymm12, xmm0
        vextractf128 xmm13, ymm0, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 320]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 352]
        vmovupd yword ptr [r11 + 320], ymm12
        vmovupd yword ptr [r11 + 352], ymm13
        vcvtps2pd ymm14, xmm1
        vextractf128 xmm15, ymm1, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 384]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 416]
        vmovupd yword ptr [r11 + 384], ymm14
        vmovupd yword ptr [r11 + 416], ymm15
        vcvtps2pd ymm12, xmm2
        vextractf128 xmm13, ymm2, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 576]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 608]
        vmovupd yword ptr [r11 + 576], ymm12
        vmovupd yword ptr [r11 + 608], ymm13
        vcvtps2pd ymm14, xmm3
        vextractf128 xmm15, ymm3, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 640]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 672]
        vmovupd yword ptr [r11 + 640], ymm14
        vmovupd yword ptr [r11 + 672], ymm15
        vcvtps2pd ymm12, xmm4
        vextractf128 xmm13, ymm4, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 832]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 864]
        vmovupd yword ptr [r11 + 832], ymm12
        vmovupd yword ptr [r11 + 864], ymm13
        vcvtps2pd ymm14, xmm5
        vextractf128 xmm15, ymm5, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 896]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 928]
        vmovupd yword ptr [r11 + 896], ymm14
        vmovupd yword ptr [r11 + 928], ymm15
        vcvtps2pd ymm12, xmm6
        vextractf128 xmm13, ymm6, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 1088]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 1120]
        vmovupd yword ptr [r11 + 1088], ymm12
        vmovupd yword ptr [r11 + 1120], ymm13
        vcvtps2pd ymm14, xmm7
        vextractf128 xmm15, ymm7, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 1152]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 1184]
        vmovupd yword ptr [r11 + 1152], ymm14
        vmovupd yword ptr [r11 + 1184], ymm15
        vcvtps2pd ymm12, xmm8
        vextractf128 xmm13, ymm8, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 1344]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 1376]
        vmovupd yword ptr [r11 + 1344], ymm12
        vmovupd yword ptr [r11 + 1376], ymm13
        vcvtps2pd ymm14, xmm9
        vextractf128 xmm15, ymm9, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 1408]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 1440]
        vmovupd yword ptr [r11 + 1408], ymm14
        vmovupd yword ptr [r11 + 1440], ymm15
        vcvtps2pd ymm12, xmm10
        vextractf128 xmm13, ymm10, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 1600]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 1632]
        vmovupd yword ptr [r11 + 1600], ymm12
        vmovupd yword ptr [r11 + 1632], ymm13
        vcvtps2pd ymm14, xmm11
        vextractf128 xmm15, ymm11, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 1664]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 1696]
        vmovupd yword ptr [r11 + 1664], ymm14
        vmovupd yword ptr [r11 + 1696], ymm15
        vxorps  ymm12, ymm12, ymm12
        vmovups yword ptr [r11 + 1856], ymm12
        vmovups yword ptr [r11 + 1888], ymm12
        vmovups yword ptr [r11 + 1984], ymm12
        vmovups yword ptr [r11 + 2016], ymm12
        vmovups yword ptr [r11 + 2112], ymm12
        vmovups yword ptr [r11 + 2144], ymm12
        vmovups yword ptr [r11 + 2240], ymm12
        vmovups yword ptr [r11 + 2272], ymm12
        vmovups yword ptr [r11 + 2368], ymm12
        vmovups yword ptr [r11 + 2400], ymm12
        vmovups yword ptr [r11 + 2496], ymm12
        vmovups yword ptr [r11 + 2528], ymm12
        mov     edi, KGEMM_FLUSH
@next:  test    rax, rax
        jnz     @block
        add     rbx, qword ptr [r11 + 40]
        sub     r10, 1
        jnz     @tap
        // the totals rounded to single, in ymm0 .. ymm11
        vcvtpd2ps xmm0, yword ptr [r11 + 320]
        vcvtpd2ps xmm12, yword ptr [r11 + 352]
        vinsertf128 ymm0, ymm0, xmm12, 1
        vcvtpd2ps xmm1, yword ptr [r11 + 384]
        vcvtpd2ps xmm12, yword ptr [r11 + 416]
        vinsertf128 ymm1, ymm1, xmm12, 1
        vcvtpd2ps xmm2, yword ptr [r11 + 576]
        vcvtpd2ps xmm12, yword ptr [r11 + 608]
        vinsertf128 ymm2, ymm2, xmm12, 1
        vcvtpd2ps xmm3, yword ptr [r11 + 640]
        vcvtpd2ps xmm12, yword ptr [r11 + 672]
        vinsertf128 ymm3, ymm3, xmm12, 1
        vcvtpd2ps xmm4, yword ptr [r11 + 832]
        vcvtpd2ps xmm12, yword ptr [r11 + 864]
        vinsertf128 ymm4, ymm4, xmm12, 1
        vcvtpd2ps xmm5, yword ptr [r11 + 896]
        vcvtpd2ps xmm12, yword ptr [r11 + 928]
        vinsertf128 ymm5, ymm5, xmm12, 1
        vcvtpd2ps xmm6, yword ptr [r11 + 1088]
        vcvtpd2ps xmm12, yword ptr [r11 + 1120]
        vinsertf128 ymm6, ymm6, xmm12, 1
        vcvtpd2ps xmm7, yword ptr [r11 + 1152]
        vcvtpd2ps xmm12, yword ptr [r11 + 1184]
        vinsertf128 ymm7, ymm7, xmm12, 1
        vcvtpd2ps xmm8, yword ptr [r11 + 1344]
        vcvtpd2ps xmm12, yword ptr [r11 + 1376]
        vinsertf128 ymm8, ymm8, xmm12, 1
        vcvtpd2ps xmm9, yword ptr [r11 + 1408]
        vcvtpd2ps xmm12, yword ptr [r11 + 1440]
        vinsertf128 ymm9, ymm9, xmm12, 1
        vcvtpd2ps xmm10, yword ptr [r11 + 1600]
        vcvtpd2ps xmm12, yword ptr [r11 + 1632]
        vinsertf128 ymm10, ymm10, xmm12, 1
        vcvtpd2ps xmm11, yword ptr [r11 + 1664]
        vcvtpd2ps xmm12, yword ptr [r11 + 1696]
        vinsertf128 ymm11, ymm11, xmm12, 1
        // the 6 rows of y: rdi = row 0, r13 = row 3, r12 = ys
        mov     rdi, qword ptr [r11 + 48]
        mov     r12, qword ptr [r11 + 56]
        lea     r13, [rdi + r12 * 2]
        add     r13, r12
        test    byte ptr [r11 + 72], 1
        jz      @store
        vaddps  ymm0, ymm0, yword ptr [rdi]
        vaddps  ymm1, ymm1, yword ptr [rdi + 32]
        vaddps  ymm2, ymm2, yword ptr [rdi + r12]
        vaddps  ymm3, ymm3, yword ptr [rdi + r12 + 32]
        vaddps  ymm4, ymm4, yword ptr [rdi + r12 * 2]
        vaddps  ymm5, ymm5, yword ptr [rdi + r12 * 2 + 32]
        vaddps  ymm6, ymm6, yword ptr [r13]
        vaddps  ymm7, ymm7, yword ptr [r13 + 32]
        vaddps  ymm8, ymm8, yword ptr [r13 + r12]
        vaddps  ymm9, ymm9, yword ptr [r13 + r12 + 32]
        vaddps  ymm10, ymm10, yword ptr [r13 + r12 * 2]
        vaddps  ymm11, ymm11, yword ptr [r13 + r12 * 2 + 32]
@store: vmovups yword ptr [rdi], ymm0
        vmovups yword ptr [rdi + 32], ymm1
        vmovups yword ptr [rdi + r12], ymm2
        vmovups yword ptr [rdi + r12 + 32], ymm3
        vmovups yword ptr [rdi + r12 * 2], ymm4
        vmovups yword ptr [rdi + r12 * 2 + 32], ymm5
        vmovups yword ptr [r13], ymm6
        vmovups yword ptr [r13 + 32], ymm7
        vmovups yword ptr [r13 + r12], ymm8
        vmovups yword ptr [r13 + r12 + 32], ymm9
        vmovups yword ptr [r13 + r12 * 2], ymm10
        vmovups yword ptr [r13 + r12 * 2 + 32], ymm11
        vzeroupper
        {$ifdef ABIWINX64}
        movups  xmm6, oword ptr [r11 + 160]
        movups  xmm7, oword ptr [r11 + 176]
        movups  xmm8, oword ptr [r11 + 192]
        movups  xmm9, oword ptr [r11 + 208]
        movups  xmm10, oword ptr [r11 + 224]
        movups  xmm11, oword ptr [r11 + 240]
        movups  xmm12, oword ptr [r11 + 256]
        movups  xmm13, oword ptr [r11 + 272]
        movups  xmm14, oword ptr [r11 + 288]
        movups  xmm15, oword ptr [r11 + 304]
        {$endif ABIWINX64}
        mov     rbx, qword ptr [r11 + 96]
        mov     rsi, qword ptr [r11 + 104]
        mov     rdi, qword ptr [r11 + 112]
        mov     r12, qword ptr [r11 + 120]
        mov     r13, qword ptr [r11 + 128]
end;

/// 1 row x 16 outputs, the leftover rows of a block: the same sums, in
// the same order, as KGemm6Avx2; rdx = the blocks before the move to double
procedure KGemm1Avx2(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        mov     qword ptr [r11 + 96], rbx
        mov     qword ptr [r11 + 104], rsi
        {$ifdef ABIWINX64}
        vmovups oword ptr [r11 + 256], xmm12
        vmovups oword ptr [r11 + 272], xmm13
        vmovups oword ptr [r11 + 288], xmm14
        vmovups oword ptr [r11 + 304], xmm15
        {$endif ABIWINX64}
        mov     rax, qword ptr [r11 + 64]
        vcvtps2pd ymm0, oword ptr [rax]
        vcvtps2pd ymm1, oword ptr [rax + 16]
        vcvtps2pd ymm2, oword ptr [rax + 32]
        vcvtps2pd ymm3, oword ptr [rax + 48]
        vmovupd yword ptr [r11 + 320], ymm0
        vmovupd yword ptr [r11 + 352], ymm1
        vmovupd yword ptr [r11 + 384], ymm2
        vmovupd yword ptr [r11 + 416], ymm3
        vxorps  ymm4, ymm4, ymm4
        vmovups yword ptr [r11 + 1856], ymm4
        vmovups yword ptr [r11 + 1888], ymm4
        mov     rbx, qword ptr [r11]
        mov     r9, qword ptr [r11 + 16]
        mov     r10, qword ptr [r11 + 32]
@tap:   mov     rsi, rbx
        mov     rax, qword ptr [r11 + 24]
        mov     edx, KGEMM_FLUSH
@block: mov     rcx, qword ptr [r11 + 80]
        cmp     rax, rcx
        cmovb   rcx, rax
        sub     rax, rcx
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
@c:     vmovups ymm12, yword ptr [r9]
        vmovups ymm13, yword ptr [r9 + 32]
        vbroadcastss ymm14, dword ptr [rsi]
        vfmadd231ps ymm0, ymm14, ymm12
        vfmadd231ps ymm1, ymm14, ymm13
        add     rsi, 4
        add     r9, 64
        sub     rcx, 1
        jnz     @c
        vaddps  ymm0, ymm0, yword ptr [r11 + 1856]
        vaddps  ymm1, ymm1, yword ptr [r11 + 1888]
        sub     edx, 1
        jz      @flush
        test    rax, rax
        jz      @flush
        vmovups yword ptr [r11 + 1856], ymm0
        vmovups yword ptr [r11 + 1888], ymm1
        jmp     @next
@flush:
        vcvtps2pd ymm12, xmm0
        vextractf128 xmm13, ymm0, 1
        vcvtps2pd ymm13, xmm13
        vaddpd  ymm12, ymm12, yword ptr [r11 + 320]
        vaddpd  ymm13, ymm13, yword ptr [r11 + 352]
        vmovupd yword ptr [r11 + 320], ymm12
        vmovupd yword ptr [r11 + 352], ymm13
        vcvtps2pd ymm14, xmm1
        vextractf128 xmm15, ymm1, 1
        vcvtps2pd ymm15, xmm15
        vaddpd  ymm14, ymm14, yword ptr [r11 + 384]
        vaddpd  ymm15, ymm15, yword ptr [r11 + 416]
        vmovupd yword ptr [r11 + 384], ymm14
        vmovupd yword ptr [r11 + 416], ymm15
        vxorps  ymm12, ymm12, ymm12
        vmovups yword ptr [r11 + 1856], ymm12
        vmovups yword ptr [r11 + 1888], ymm12
        mov     edx, KGEMM_FLUSH
@next:  test    rax, rax
        jnz     @block
        add     rbx, qword ptr [r11 + 40]
        sub     r10, 1
        jnz     @tap
        vcvtpd2ps xmm0, yword ptr [r11 + 320]
        vcvtpd2ps xmm12, yword ptr [r11 + 352]
        vinsertf128 ymm0, ymm0, xmm12, 1
        vcvtpd2ps xmm1, yword ptr [r11 + 384]
        vcvtpd2ps xmm12, yword ptr [r11 + 416]
        vinsertf128 ymm1, ymm1, xmm12, 1
        mov     rax, qword ptr [r11 + 48]
        test    byte ptr [r11 + 72], 1
        jz      @store
        vaddps  ymm0, ymm0, yword ptr [rax]
        vaddps  ymm1, ymm1, yword ptr [rax + 32]
@store: vmovups yword ptr [rax], ymm0
        vmovups yword ptr [rax + 32], ymm1
        vzeroupper
        {$ifdef ABIWINX64}
        movups  xmm12, oword ptr [r11 + 256]
        movups  xmm13, oword ptr [r11 + 272]
        movups  xmm14, oword ptr [r11 + 288]
        movups  xmm15, oword ptr [r11 + 304]
        {$endif ABIWINX64}
        mov     rbx, qword ptr [r11 + 96]
        mov     rsi, qword ptr [r11 + 104]
end;

/// 6 rows x 32 outputs (two panels, AVX-512): the same sums, in the same
// order, as KGemm6Avx2 on each of the two panels (the same bits)
// - zmm0 .. zmm11 the accumulators (row i: zmm2i the panel, zmm2i+1 the
// next one), zmm12 / zmm13 the weights of both panels, zmm14 / zmm15 the
// broadcast inputs, zmm16 .. zmm27 the intermediate totals, zmm28 .. zmm31
// the conversions to double; r14 = the gap from one panel to the next (bytes)
procedure KGemm6x32Avx512(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        mov     qword ptr [r11 + 96], rbx
        mov     qword ptr [r11 + 104], rsi
        mov     qword ptr [r11 + 112], rdi
        mov     qword ptr [r11 + 120], r12
        mov     qword ptr [r11 + 128], r13
        mov     qword ptr [r11 + 136], r14
        {$ifdef ABIWINX64}
        vmovups oword ptr [r11 + 160], xmm6
        vmovups oword ptr [r11 + 176], xmm7
        vmovups oword ptr [r11 + 192], xmm8
        vmovups oword ptr [r11 + 208], xmm9
        vmovups oword ptr [r11 + 224], xmm10
        vmovups oword ptr [r11 + 240], xmm11
        vmovups oword ptr [r11 + 256], xmm12
        vmovups oword ptr [r11 + 272], xmm13
        vmovups oword ptr [r11 + 288], xmm14
        vmovups oword ptr [r11 + 304], xmm15
        {$endif ABIWINX64}
        // the totals start from the bias of both panels, in double
        mov     rax, qword ptr [r11 + 64]
        vmovups zmm16, [rax]
        vmovups zmm17, [rax + 64]
        vcvtps2pd zmm20, ymm16
        vextractf64x4 ymm16, zmm16, 1
        vcvtps2pd zmm21, ymm16
        vcvtps2pd zmm22, ymm17
        vextractf64x4 ymm17, zmm17, 1
        vcvtps2pd zmm23, ymm17
        vmovupd [r11 + 320], zmm20
        vmovupd [r11 + 384], zmm21
        vmovupd [r11 + 448], zmm22
        vmovupd [r11 + 512], zmm23
        vmovupd [r11 + 576], zmm20
        vmovupd [r11 + 640], zmm21
        vmovupd [r11 + 704], zmm22
        vmovupd [r11 + 768], zmm23
        vmovupd [r11 + 832], zmm20
        vmovupd [r11 + 896], zmm21
        vmovupd [r11 + 960], zmm22
        vmovupd [r11 + 1024], zmm23
        vmovupd [r11 + 1088], zmm20
        vmovupd [r11 + 1152], zmm21
        vmovupd [r11 + 1216], zmm22
        vmovupd [r11 + 1280], zmm23
        vmovupd [r11 + 1344], zmm20
        vmovupd [r11 + 1408], zmm21
        vmovupd [r11 + 1472], zmm22
        vmovupd [r11 + 1536], zmm23
        vmovupd [r11 + 1600], zmm20
        vmovupd [r11 + 1664], zmm21
        vmovupd [r11 + 1728], zmm22
        vmovupd [r11 + 1792], zmm23
        vpxord  zmm16, zmm16, zmm16
        vpxord  zmm17, zmm17, zmm17
        vpxord  zmm18, zmm18, zmm18
        vpxord  zmm19, zmm19, zmm19
        vpxord  zmm20, zmm20, zmm20
        vpxord  zmm21, zmm21, zmm21
        vpxord  zmm22, zmm22, zmm22
        vpxord  zmm23, zmm23, zmm23
        vpxord  zmm24, zmm24, zmm24
        vpxord  zmm25, zmm25, zmm25
        vpxord  zmm26, zmm26, zmm26
        vpxord  zmm27, zmm27, zmm27
        mov     rbx, qword ptr [r11]
        mov     rdx, qword ptr [r11 + 8]
        mov     r9, qword ptr [r11 + 16]
        mov     r10, qword ptr [r11 + 32]
        mov     r14, qword ptr [r11 + 88]
@tap:   mov     rsi, rbx
        lea     r8, [rbx + rdx * 2]
        add     r8, rdx
        mov     rax, qword ptr [r11 + 24]
        mov     edi, KGEMM_FLUSH
@block: mov     rcx, qword ptr [r11 + 80]
        cmp     rax, rcx
        cmovb   rcx, rax
        sub     rax, rcx
        vpxord  zmm0, zmm0, zmm0
        vpxord  zmm1, zmm1, zmm1
        vpxord  zmm2, zmm2, zmm2
        vpxord  zmm3, zmm3, zmm3
        vpxord  zmm4, zmm4, zmm4
        vpxord  zmm5, zmm5, zmm5
        vpxord  zmm6, zmm6, zmm6
        vpxord  zmm7, zmm7, zmm7
        vpxord  zmm8, zmm8, zmm8
        vpxord  zmm9, zmm9, zmm9
        vpxord  zmm10, zmm10, zmm10
        vpxord  zmm11, zmm11, zmm11
@c:     vmovups zmm12, [r9]
        vmovups zmm13, [r9 + r14]
        vbroadcastss zmm14, dword ptr [rsi]
        vbroadcastss zmm15, dword ptr [rsi + rdx]
        vfmadd231ps zmm0, zmm14, zmm12
        vfmadd231ps zmm1, zmm14, zmm13
        vfmadd231ps zmm2, zmm15, zmm12
        vfmadd231ps zmm3, zmm15, zmm13
        vbroadcastss zmm14, dword ptr [rsi + rdx * 2]
        vbroadcastss zmm15, dword ptr [r8]
        vfmadd231ps zmm4, zmm14, zmm12
        vfmadd231ps zmm5, zmm14, zmm13
        vfmadd231ps zmm6, zmm15, zmm12
        vfmadd231ps zmm7, zmm15, zmm13
        vbroadcastss zmm14, dword ptr [r8 + rdx]
        vbroadcastss zmm15, dword ptr [r8 + rdx * 2]
        vfmadd231ps zmm8, zmm14, zmm12
        vfmadd231ps zmm9, zmm14, zmm13
        vfmadd231ps zmm10, zmm15, zmm12
        vfmadd231ps zmm11, zmm15, zmm13
        add     rsi, 4
        add     r8, 4
        add     r9, 64
        sub     rcx, 1
        jnz     @c
        // the block is added to the intermediate total, in the registers
        vaddps  zmm16, zmm0, zmm16
        vaddps  zmm17, zmm1, zmm17
        vaddps  zmm18, zmm2, zmm18
        vaddps  zmm19, zmm3, zmm19
        vaddps  zmm20, zmm4, zmm20
        vaddps  zmm21, zmm5, zmm21
        vaddps  zmm22, zmm6, zmm22
        vaddps  zmm23, zmm7, zmm23
        vaddps  zmm24, zmm8, zmm24
        vaddps  zmm25, zmm9, zmm25
        vaddps  zmm26, zmm10, zmm26
        vaddps  zmm27, zmm11, zmm27
        sub     edi, 1
        jz      @flush
        test    rax, rax
        jnz     @block
        // every KGEMM_FLUSH blocks and at the end of a tap: into the total in
        // double, then the intermediate one back to zero
@flush:
        vcvtps2pd zmm28, ymm16
        vextractf64x4 ymm29, zmm16, 1
        vcvtps2pd zmm29, ymm29
        vaddpd  zmm28, zmm28, [r11 + 320]
        vaddpd  zmm29, zmm29, [r11 + 384]
        vmovupd [r11 + 320], zmm28
        vmovupd [r11 + 384], zmm29
        vcvtps2pd zmm30, ymm17
        vextractf64x4 ymm31, zmm17, 1
        vcvtps2pd zmm31, ymm31
        vaddpd  zmm30, zmm30, [r11 + 448]
        vaddpd  zmm31, zmm31, [r11 + 512]
        vmovupd [r11 + 448], zmm30
        vmovupd [r11 + 512], zmm31
        vcvtps2pd zmm28, ymm18
        vextractf64x4 ymm29, zmm18, 1
        vcvtps2pd zmm29, ymm29
        vaddpd  zmm28, zmm28, [r11 + 576]
        vaddpd  zmm29, zmm29, [r11 + 640]
        vmovupd [r11 + 576], zmm28
        vmovupd [r11 + 640], zmm29
        vcvtps2pd zmm30, ymm19
        vextractf64x4 ymm31, zmm19, 1
        vcvtps2pd zmm31, ymm31
        vaddpd  zmm30, zmm30, [r11 + 704]
        vaddpd  zmm31, zmm31, [r11 + 768]
        vmovupd [r11 + 704], zmm30
        vmovupd [r11 + 768], zmm31
        vcvtps2pd zmm28, ymm20
        vextractf64x4 ymm29, zmm20, 1
        vcvtps2pd zmm29, ymm29
        vaddpd  zmm28, zmm28, [r11 + 832]
        vaddpd  zmm29, zmm29, [r11 + 896]
        vmovupd [r11 + 832], zmm28
        vmovupd [r11 + 896], zmm29
        vcvtps2pd zmm30, ymm21
        vextractf64x4 ymm31, zmm21, 1
        vcvtps2pd zmm31, ymm31
        vaddpd  zmm30, zmm30, [r11 + 960]
        vaddpd  zmm31, zmm31, [r11 + 1024]
        vmovupd [r11 + 960], zmm30
        vmovupd [r11 + 1024], zmm31
        vcvtps2pd zmm28, ymm22
        vextractf64x4 ymm29, zmm22, 1
        vcvtps2pd zmm29, ymm29
        vaddpd  zmm28, zmm28, [r11 + 1088]
        vaddpd  zmm29, zmm29, [r11 + 1152]
        vmovupd [r11 + 1088], zmm28
        vmovupd [r11 + 1152], zmm29
        vcvtps2pd zmm30, ymm23
        vextractf64x4 ymm31, zmm23, 1
        vcvtps2pd zmm31, ymm31
        vaddpd  zmm30, zmm30, [r11 + 1216]
        vaddpd  zmm31, zmm31, [r11 + 1280]
        vmovupd [r11 + 1216], zmm30
        vmovupd [r11 + 1280], zmm31
        vcvtps2pd zmm28, ymm24
        vextractf64x4 ymm29, zmm24, 1
        vcvtps2pd zmm29, ymm29
        vaddpd  zmm28, zmm28, [r11 + 1344]
        vaddpd  zmm29, zmm29, [r11 + 1408]
        vmovupd [r11 + 1344], zmm28
        vmovupd [r11 + 1408], zmm29
        vcvtps2pd zmm30, ymm25
        vextractf64x4 ymm31, zmm25, 1
        vcvtps2pd zmm31, ymm31
        vaddpd  zmm30, zmm30, [r11 + 1472]
        vaddpd  zmm31, zmm31, [r11 + 1536]
        vmovupd [r11 + 1472], zmm30
        vmovupd [r11 + 1536], zmm31
        vcvtps2pd zmm28, ymm26
        vextractf64x4 ymm29, zmm26, 1
        vcvtps2pd zmm29, ymm29
        vaddpd  zmm28, zmm28, [r11 + 1600]
        vaddpd  zmm29, zmm29, [r11 + 1664]
        vmovupd [r11 + 1600], zmm28
        vmovupd [r11 + 1664], zmm29
        vcvtps2pd zmm30, ymm27
        vextractf64x4 ymm31, zmm27, 1
        vcvtps2pd zmm31, ymm31
        vaddpd  zmm30, zmm30, [r11 + 1728]
        vaddpd  zmm31, zmm31, [r11 + 1792]
        vmovupd [r11 + 1728], zmm30
        vmovupd [r11 + 1792], zmm31
        vpxord  zmm16, zmm16, zmm16
        vpxord  zmm17, zmm17, zmm17
        vpxord  zmm18, zmm18, zmm18
        vpxord  zmm19, zmm19, zmm19
        vpxord  zmm20, zmm20, zmm20
        vpxord  zmm21, zmm21, zmm21
        vpxord  zmm22, zmm22, zmm22
        vpxord  zmm23, zmm23, zmm23
        vpxord  zmm24, zmm24, zmm24
        vpxord  zmm25, zmm25, zmm25
        vpxord  zmm26, zmm26, zmm26
        vpxord  zmm27, zmm27, zmm27
        mov     edi, KGEMM_FLUSH
        test    rax, rax
        jnz     @block
@tapend:
        add     rbx, qword ptr [r11 + 40]
        sub     r10, 1
        jnz     @tap
        // the totals rounded to single, in zmm0 .. zmm11
        vmovupd zmm28, [r11 + 320]
        vmovupd zmm29, [r11 + 384]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm0, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 448]
        vmovupd zmm29, [r11 + 512]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm1, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 576]
        vmovupd zmm29, [r11 + 640]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm2, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 704]
        vmovupd zmm29, [r11 + 768]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm3, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 832]
        vmovupd zmm29, [r11 + 896]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm4, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 960]
        vmovupd zmm29, [r11 + 1024]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm5, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 1088]
        vmovupd zmm29, [r11 + 1152]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm6, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 1216]
        vmovupd zmm29, [r11 + 1280]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm7, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 1344]
        vmovupd zmm29, [r11 + 1408]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm8, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 1472]
        vmovupd zmm29, [r11 + 1536]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm9, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 1600]
        vmovupd zmm29, [r11 + 1664]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm10, zmm28, ymm29, 1
        vmovupd zmm28, [r11 + 1728]
        vmovupd zmm29, [r11 + 1792]
        vcvtpd2ps ymm28, zmm28
        vcvtpd2ps ymm29, zmm29
        vinsertf64x4 zmm11, zmm28, ymm29, 1
        // the 6 rows of y: rdi = row 0, r13 = row 3, r12 = ys
        mov     rdi, qword ptr [r11 + 48]
        mov     r12, qword ptr [r11 + 56]
        lea     r13, [rdi + r12 * 2]
        add     r13, r12
        test    byte ptr [r11 + 72], 1
        jz      @store
        vaddps  zmm0, zmm0, [rdi]
        vaddps  zmm1, zmm1, [rdi + 64]
        vaddps  zmm2, zmm2, [rdi + r12]
        vaddps  zmm3, zmm3, [rdi + r12 + 64]
        vaddps  zmm4, zmm4, [rdi + r12 * 2]
        vaddps  zmm5, zmm5, [rdi + r12 * 2 + 64]
        vaddps  zmm6, zmm6, [r13]
        vaddps  zmm7, zmm7, [r13 + 64]
        vaddps  zmm8, zmm8, [r13 + r12]
        vaddps  zmm9, zmm9, [r13 + r12 + 64]
        vaddps  zmm10, zmm10, [r13 + r12 * 2]
        vaddps  zmm11, zmm11, [r13 + r12 * 2 + 64]
@store: vmovups [rdi], zmm0
        vmovups [rdi + 64], zmm1
        vmovups [rdi + r12], zmm2
        vmovups [rdi + r12 + 64], zmm3
        vmovups [rdi + r12 * 2], zmm4
        vmovups [rdi + r12 * 2 + 64], zmm5
        vmovups [r13], zmm6
        vmovups [r13 + 64], zmm7
        vmovups [r13 + r12], zmm8
        vmovups [r13 + r12 + 64], zmm9
        vmovups [r13 + r12 * 2], zmm10
        vmovups [r13 + r12 * 2 + 64], zmm11
        vzeroupper
        {$ifdef ABIWINX64}
        movups  xmm6, oword ptr [r11 + 160]
        movups  xmm7, oword ptr [r11 + 176]
        movups  xmm8, oword ptr [r11 + 192]
        movups  xmm9, oword ptr [r11 + 208]
        movups  xmm10, oword ptr [r11 + 224]
        movups  xmm11, oword ptr [r11 + 240]
        movups  xmm12, oword ptr [r11 + 256]
        movups  xmm13, oword ptr [r11 + 272]
        movups  xmm14, oword ptr [r11 + 288]
        movups  xmm15, oword ptr [r11 + 304]
        {$endif ABIWINX64}
        mov     rbx, qword ptr [r11 + 96]
        mov     rsi, qword ptr [r11 + 104]
        mov     rdi, qword ptr [r11 + 112]
        mov     r12, qword ptr [r11 + 120]
        mov     r13, qword ptr [r11 + 128]
        mov     r14, qword ptr [r11 + 136]
end;

/// KokoroLstmCell, 4 units at a time: A = gx, gh, b, c, h, hidden, the
// constants (KLstmConsts)
// - the sum of the gates in single, then all in double as Sigmoid and TanhS
// (System's Exp is in double): exp(x) = 2^n p(r), n = round(x / ln 2),
// r = x - n ln 2 in two pieces, p Taylor at degree 13 (|r| <= ln 2 / 2:
// relative error below 1e-17); rounded to single at the end as theirs
// - rsi / rdi / r8 = gx / gh / b at unit j, r10 = H x 4 bytes (from one gate
// to the next), r9 = 3 H x 4, rbx = c, r12 = h, rdx = the groups of 4
// units left, rax = the constants; xmm6 .. xmm9 the gates i, f, g, o
procedure KLstmCellAvx2(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        mov     qword ptr [r11 + 96], rbx
        mov     qword ptr [r11 + 104], rsi
        mov     qword ptr [r11 + 112], rdi
        mov     qword ptr [r11 + 120], r12
        {$ifdef ABIWINX64}
        vmovups oword ptr [r11 + 160], xmm6
        vmovups oword ptr [r11 + 176], xmm7
        vmovups oword ptr [r11 + 192], xmm8
        vmovups oword ptr [r11 + 208], xmm9
        vmovups oword ptr [r11 + 224], xmm10
        vmovups oword ptr [r11 + 240], xmm11
        {$endif ABIWINX64}
        mov     rsi, qword ptr [r11]
        mov     rdi, qword ptr [r11 + 8]
        mov     r8, qword ptr [r11 + 16]
        mov     rbx, qword ptr [r11 + 24]
        mov     r12, qword ptr [r11 + 32]
        mov     rdx, qword ptr [r11 + 40]
        mov     rax, qword ptr [r11 + 48]
        lea     r10, [rdx * 4]
        lea     r9, [r10 + r10 * 2]
        shr     rdx, 2
        jz      @done
@unit:
        // gate 0: sigmoid
        vmovups xmm1, oword ptr [rsi]
        vaddps  xmm1, xmm1, oword ptr [rdi]
        vaddps  xmm1, xmm1, oword ptr [r8]
        vcvtps2pd ymm1, xmm1
        vxorpd  ymm1, ymm1, yword ptr [rax + 576]
        vmaxpd  ymm1, ymm1, yword ptr [rax + 672]
        vminpd  ymm1, ymm1, yword ptr [rax + 640]
        vmulpd  ymm3, ymm1, yword ptr [rax + 0]
        vroundpd ymm3, ymm3, 0
        vmovapd ymm4, ymm1
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 32]
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 64]
        vmovupd ymm2, yword ptr [rax + 512]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 480]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 448]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 416]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 384]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 352]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 320]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 288]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 256]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 224]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 192]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 160]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 128]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 96]
        vcvtpd2dq xmm5, ymm3
        vpmovsxdq ymm5, xmm5
        vpaddq  ymm5, ymm5, yword ptr [rax + 608]
        vpsllq  ymm5, ymm5, 52
        vmulpd  ymm2, ymm2, ymm5
        vaddpd  ymm2, ymm2, yword ptr [rax + 544]
        vmovupd ymm3, yword ptr [rax + 544]
        vdivpd  ymm2, ymm3, ymm2
        vcvtpd2ps xmm6, ymm2
        // gate 1: sigmoid
        vmovups xmm1, oword ptr [rsi + r10]
        vaddps  xmm1, xmm1, oword ptr [rdi + r10]
        vaddps  xmm1, xmm1, oword ptr [r8 + r10]
        vcvtps2pd ymm1, xmm1
        vxorpd  ymm1, ymm1, yword ptr [rax + 576]
        vmaxpd  ymm1, ymm1, yword ptr [rax + 672]
        vminpd  ymm1, ymm1, yword ptr [rax + 640]
        vmulpd  ymm3, ymm1, yword ptr [rax + 0]
        vroundpd ymm3, ymm3, 0
        vmovapd ymm4, ymm1
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 32]
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 64]
        vmovupd ymm2, yword ptr [rax + 512]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 480]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 448]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 416]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 384]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 352]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 320]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 288]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 256]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 224]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 192]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 160]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 128]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 96]
        vcvtpd2dq xmm5, ymm3
        vpmovsxdq ymm5, xmm5
        vpaddq  ymm5, ymm5, yword ptr [rax + 608]
        vpsllq  ymm5, ymm5, 52
        vmulpd  ymm2, ymm2, ymm5
        vaddpd  ymm2, ymm2, yword ptr [rax + 544]
        vmovupd ymm3, yword ptr [rax + 544]
        vdivpd  ymm2, ymm3, ymm2
        vcvtpd2ps xmm7, ymm2
        // gate 2: hyperbolic tangent
        vmovups xmm1, oword ptr [rsi + r10 * 2]
        vaddps  xmm1, xmm1, oword ptr [rdi + r10 * 2]
        vaddps  xmm1, xmm1, oword ptr [r8 + r10 * 2]
        vmovaps xmm10, xmm1
        vcvtps2pd ymm1, xmm10
        vmaxpd  ymm1, ymm1, yword ptr [rax + 736]
        vminpd  ymm1, ymm1, yword ptr [rax + 704]
        vaddpd  ymm1, ymm1, ymm1
        vmulpd  ymm3, ymm1, yword ptr [rax + 0]
        vroundpd ymm3, ymm3, 0
        vmovapd ymm4, ymm1
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 32]
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 64]
        vmovupd ymm2, yword ptr [rax + 512]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 480]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 448]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 416]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 384]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 352]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 320]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 288]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 256]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 224]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 192]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 160]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 128]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 96]
        vcvtpd2dq xmm5, ymm3
        vpmovsxdq ymm5, xmm5
        vpaddq  ymm5, ymm5, yword ptr [rax + 608]
        vpsllq  ymm5, ymm5, 52
        vmulpd  ymm2, ymm2, ymm5
        vsubpd  ymm3, ymm2, yword ptr [rax + 544]
        vaddpd  ymm2, ymm2, yword ptr [rax + 544]
        vdivpd  ymm2, ymm3, ymm2
        vcvtpd2ps xmm8, ymm2
        // gate 3: sigmoid
        vmovups xmm1, oword ptr [rsi + r9]
        vaddps  xmm1, xmm1, oword ptr [rdi + r9]
        vaddps  xmm1, xmm1, oword ptr [r8 + r9]
        vcvtps2pd ymm1, xmm1
        vxorpd  ymm1, ymm1, yword ptr [rax + 576]
        vmaxpd  ymm1, ymm1, yword ptr [rax + 672]
        vminpd  ymm1, ymm1, yword ptr [rax + 640]
        vmulpd  ymm3, ymm1, yword ptr [rax + 0]
        vroundpd ymm3, ymm3, 0
        vmovapd ymm4, ymm1
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 32]
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 64]
        vmovupd ymm2, yword ptr [rax + 512]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 480]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 448]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 416]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 384]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 352]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 320]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 288]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 256]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 224]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 192]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 160]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 128]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 96]
        vcvtpd2dq xmm5, ymm3
        vpmovsxdq ymm5, xmm5
        vpaddq  ymm5, ymm5, yword ptr [rax + 608]
        vpsllq  ymm5, ymm5, 52
        vmulpd  ymm2, ymm2, ymm5
        vaddpd  ymm2, ymm2, yword ptr [rax + 544]
        vmovupd ymm3, yword ptr [rax + 544]
        vdivpd  ymm2, ymm3, ymm2
        vcvtpd2ps xmm9, ymm2
        // c := f * c + i * g; h := o * TanhS(c), in single
        vmulps  xmm10, xmm7, oword ptr [rbx]
        vmulps  xmm11, xmm6, xmm8
        vaddps  xmm10, xmm10, xmm11
        vmovups oword ptr [rbx], xmm10
        vcvtps2pd ymm1, xmm10
        vmaxpd  ymm1, ymm1, yword ptr [rax + 736]
        vminpd  ymm1, ymm1, yword ptr [rax + 704]
        vaddpd  ymm1, ymm1, ymm1
        vmulpd  ymm3, ymm1, yword ptr [rax + 0]
        vroundpd ymm3, ymm3, 0
        vmovapd ymm4, ymm1
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 32]
        vfnmadd231pd ymm4, ymm3, yword ptr [rax + 64]
        vmovupd ymm2, yword ptr [rax + 512]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 480]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 448]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 416]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 384]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 352]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 320]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 288]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 256]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 224]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 192]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 160]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 128]
        vfmadd213pd ymm2, ymm4, yword ptr [rax + 96]
        vcvtpd2dq xmm5, ymm3
        vpmovsxdq ymm5, xmm5
        vpaddq  ymm5, ymm5, yword ptr [rax + 608]
        vpsllq  ymm5, ymm5, 52
        vmulpd  ymm2, ymm2, ymm5
        vsubpd  ymm3, ymm2, yword ptr [rax + 544]
        vaddpd  ymm2, ymm2, yword ptr [rax + 544]
        vdivpd  ymm2, ymm3, ymm2
        vcvtpd2ps xmm11, ymm2
        vmulps  xmm11, xmm9, xmm11
        vmovups oword ptr [r12], xmm11
        add     rsi, 16
        add     rdi, 16
        add     r8, 16
        add     rbx, 16
        add     r12, 16
        sub     rdx, 1
        jnz     @unit
@done:  vzeroupper
        {$ifdef ABIWINX64}
        movups  xmm6, oword ptr [r11 + 160]
        movups  xmm7, oword ptr [r11 + 176]
        movups  xmm8, oword ptr [r11 + 192]
        movups  xmm9, oword ptr [r11 + 208]
        movups  xmm10, oword ptr [r11 + 224]
        movups  xmm11, oword ptr [r11 + 240]
        {$endif ABIWINX64}
        mov     rbx, qword ptr [r11 + 96]
        mov     rsi, qword ptr [r11 + 104]
        mov     rdi, qword ptr [r11 + 112]
        mov     r12, qword ptr [r11 + 120]
end;

/// KokoroColSums, columns by 16: A = x, rows, cols16, stride (bytes), part
procedure KColSumsAvx2(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        mov     r8, qword ptr [r11]
        mov     rdx, qword ptr [r11 + 16]
        mov     r10, qword ptr [r11 + 24]
        mov     r9, qword ptr [r11 + 32]
        test    rdx, rdx
        jz      @done
        cmp     qword ptr [r11 + 8], 0
        je      @done
@chunk: vmovupd ymm0, yword ptr [r9]
        vmovupd ymm1, yword ptr [r9 + 32]
        vmovupd ymm2, yword ptr [r9 + 64]
        vmovupd ymm3, yword ptr [r9 + 96]
        mov     rax, r8
        mov     rcx, qword ptr [r11 + 8]
@row:   vcvtps2pd ymm4, oword ptr [rax]
        vaddpd  ymm0, ymm0, ymm4
        vcvtps2pd ymm5, oword ptr [rax + 16]
        vaddpd  ymm1, ymm1, ymm5
        vcvtps2pd ymm4, oword ptr [rax + 32]
        vaddpd  ymm2, ymm2, ymm4
        vcvtps2pd ymm5, oword ptr [rax + 48]
        vaddpd  ymm3, ymm3, ymm5
        add     rax, r10
        sub     rcx, 1
        jnz     @row
        vmovupd yword ptr [r9], ymm0
        vmovupd yword ptr [r9 + 32], ymm1
        vmovupd yword ptr [r9 + 64], ymm2
        vmovupd yword ptr [r9 + 96], ymm3
        add     r8, 64
        add     r9, 128
        sub     rdx, 16
        jnz     @chunk
@done:  vzeroupper
end;

/// KokoroColSquares, columns by 16: A = x, rows, cols16, stride (bytes),
// part, mean; d = x - mean, part + d * d in double, no FMA
procedure KColSquaresAvx2(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        {$ifdef ABIWINX64}
        vmovups oword ptr [r11 + 160], xmm6
        vmovups oword ptr [r11 + 176], xmm7
        vmovups oword ptr [r11 + 192], xmm8
        vmovups oword ptr [r11 + 208], xmm9
        {$endif ABIWINX64}
        mov     r8, qword ptr [r11]
        mov     rdx, qword ptr [r11 + 16]
        mov     r10, qword ptr [r11 + 24]
        mov     r9, qword ptr [r11 + 32]
        test    rdx, rdx
        jz      @done
        cmp     qword ptr [r11 + 8], 0
        je      @done
@chunk: vmovupd ymm0, yword ptr [r9]
        vmovupd ymm1, yword ptr [r9 + 32]
        vmovupd ymm2, yword ptr [r9 + 64]
        vmovupd ymm3, yword ptr [r9 + 96]
        mov     rax, qword ptr [r11 + 40]
        vmovupd ymm4, yword ptr [rax]
        vmovupd ymm5, yword ptr [rax + 32]
        vmovupd ymm6, yword ptr [rax + 64]
        vmovupd ymm7, yword ptr [rax + 96]
        add     qword ptr [r11 + 40], 128
        mov     rax, r8
        mov     rcx, qword ptr [r11 + 8]
@row:   vcvtps2pd ymm8, oword ptr [rax]
        vsubpd  ymm8, ymm8, ymm4
        vmulpd  ymm8, ymm8, ymm8
        vaddpd  ymm0, ymm0, ymm8
        vcvtps2pd ymm9, oword ptr [rax + 16]
        vsubpd  ymm9, ymm9, ymm5
        vmulpd  ymm9, ymm9, ymm9
        vaddpd  ymm1, ymm1, ymm9
        vcvtps2pd ymm8, oword ptr [rax + 32]
        vsubpd  ymm8, ymm8, ymm6
        vmulpd  ymm8, ymm8, ymm8
        vaddpd  ymm2, ymm2, ymm8
        vcvtps2pd ymm9, oword ptr [rax + 48]
        vsubpd  ymm9, ymm9, ymm7
        vmulpd  ymm9, ymm9, ymm9
        vaddpd  ymm3, ymm3, ymm9
        add     rax, r10
        sub     rcx, 1
        jnz     @row
        vmovupd yword ptr [r9], ymm0
        vmovupd yword ptr [r9 + 32], ymm1
        vmovupd yword ptr [r9 + 64], ymm2
        vmovupd yword ptr [r9 + 96], ymm3
        add     r8, 64
        add     r9, 128
        sub     rdx, 16
        jnz     @chunk
@done:  vzeroupper
        {$ifdef ABIWINX64}
        movups  xmm6, oword ptr [r11 + 160]
        movups  xmm7, oword ptr [r11 + 176]
        movups  xmm8, oword ptr [r11 + 192]
        movups  xmm9, oword ptr [r11 + 208]
        {$endif ABIWINX64}
end;

/// KokoroAffineSnake, columns by 8: A = x, rows, cols8, xs (bytes), y, ys
// (bytes), scale, shift, alpha, inv, the constants (KSnakeConsts)
// - KokoroSin as is: n = sign(q) * trunc(|q| + 1/2) (vpsignd), the two
// polynomials, then the choice by n and 1 and the sign by n and 2
// - rsi = scale, rdi = shift, rbx = alpha, r12 = inv, r10 = the constants,
// r8 / r9 = the rows of x / y, rax = the column (bytes), rdx = the columns
// left, rcx = the rows left; ymm0 .. ymm5 only
procedure KSnakeAvx2(var aArgs: TKGemmArgs);
  {$ifdef FPC} nostackframe; assembler; asm {$else} asm .noframe {$endif}
        {$ifdef ABIWINX64}
        mov     r11, rcx
        {$else}
        mov     r11, rdi
        {$endif ABIWINX64}
        mov     qword ptr [r11 + 96], rbx
        mov     qword ptr [r11 + 104], rsi
        mov     qword ptr [r11 + 112], rdi
        mov     qword ptr [r11 + 120], r12
        mov     r8, qword ptr [r11]
        mov     rcx, qword ptr [r11 + 8]
        mov     r9, qword ptr [r11 + 32]
        mov     rsi, qword ptr [r11 + 48]
        mov     rdi, qword ptr [r11 + 56]
        mov     rbx, qword ptr [r11 + 64]
        mov     r12, qword ptr [r11 + 72]
        mov     r10, qword ptr [r11 + 80]
        test    rcx, rcx
        jz      @done
@row:   xor     eax, eax
        mov     rdx, qword ptr [r11 + 16]
        test    rdx, rdx
        jz      @next
@col:   vmovups ymm0, yword ptr [r8 + rax]
        test    rsi, rsi
        jz      @snake
        // AdaIN: x * scale + shift
        vmulps  ymm0, ymm0, yword ptr [rsi + rax]
        vaddps  ymm0, ymm0, yword ptr [rdi + rax]
@snake: test    rbx, rbx
        jz      @store
        // v = alpha * x, q = v * 2 / pi, n = sign(q) * trunc(|q| + 1/2)
        vmulps  ymm1, ymm0, yword ptr [rbx + rax]
        vmulps  ymm2, ymm1, yword ptr [r10]
        vandps  ymm3, ymm2, yword ptr [r10 + 384]
        vaddps  ymm3, ymm3, yword ptr [r10 + 32]
        vcvttps2dq ymm3, ymm3
        vpsignd ymm3, ymm3, ymm2
        // r = v - n * pi / 2, in three pieces
        vcvtdq2ps ymm4, ymm3
        vmulps  ymm5, ymm4, yword ptr [r10 + 64]
        vsubps  ymm1, ymm1, ymm5
        vmulps  ymm5, ymm4, yword ptr [r10 + 96]
        vsubps  ymm1, ymm1, ymm5
        vmulps  ymm5, ymm4, yword ptr [r10 + 128]
        vsubps  ymm1, ymm1, ymm5
        // z = r * r; s = r + r * z * (S1 + z * (S2 + z * S3))
        vmulps  ymm2, ymm1, ymm1
        vmulps  ymm4, ymm2, yword ptr [r10 + 224]
        vaddps  ymm4, ymm4, yword ptr [r10 + 192]
        vmulps  ymm4, ymm4, ymm2
        vaddps  ymm4, ymm4, yword ptr [r10 + 160]
        vmulps  ymm5, ymm1, ymm2
        vmulps  ymm5, ymm5, ymm4
        vaddps  ymm5, ymm5, ymm1
        // c = 1 - z / 2 + z * z * (C1 + z * (C2 + z * C3))
        vmulps  ymm1, ymm2, yword ptr [r10 + 32]
        vmovups ymm4, yword ptr [r10 + 352]
        vsubps  ymm4, ymm4, ymm1
        vmulps  ymm1, ymm2, yword ptr [r10 + 320]
        vaddps  ymm1, ymm1, yword ptr [r10 + 288]
        vmulps  ymm1, ymm1, ymm2
        vaddps  ymm1, ymm1, yword ptr [r10 + 256]
        vmulps  ymm2, ymm2, ymm2
        vmulps  ymm1, ymm2, ymm1
        vaddps  ymm1, ymm4, ymm1
        // n odd: c, otherwise s; n and 2: the opposite
        vpand   ymm4, ymm3, yword ptr [r10 + 448]
        vpcmpeqd ymm4, ymm4, yword ptr [r10 + 448]
        vandps  ymm1, ymm1, ymm4
        vandnps ymm4, ymm4, ymm5
        vorps   ymm5, ymm1, ymm4
        vpand   ymm4, ymm3, yword ptr [r10 + 480]
        vpcmpeqd ymm4, ymm4, yword ptr [r10 + 480]
        vandps  ymm4, ymm4, yword ptr [r10 + 416]
        vxorps  ymm5, ymm5, ymm4
        // x + inv * (s * s)
        vmulps  ymm5, ymm5, ymm5
        vmulps  ymm5, ymm5, yword ptr [r12 + rax]
        vaddps  ymm0, ymm0, ymm5
@store: vmovups yword ptr [r9 + rax], ymm0
        add     rax, 32
        sub     rdx, 8
        jnz     @col
@next:  add     r8, qword ptr [r11 + 24]
        add     r9, qword ptr [r11 + 40]
        sub     rcx, 1
        jnz     @row
@done:  vzeroupper
        mov     rbx, qword ptr [r11 + 96]
        mov     rsi, qword ptr [r11 + 104]
        mov     rdi, qword ptr [r11 + 112]
        mov     r12, qword ptr [r11 + 120]
end;

{$endif ASMX64AVX1}
{$endif CPUX64}

procedure KokoroPack(var m: TKPanels; w: PSingleArray; aOut, aCin, aTaps: PtrInt;
  b: PSingleArray);
var
  p, j, o, k, c: PtrInt;
  dst: PSingleArray;
begin
  m.Cin := aCin;
  m.Taps := aTaps;
  m.Ktot := aCin * aTaps;
  m.OutCount := aOut;
  m.Panels := (aOut + KGEMM_NR - 1) div KGEMM_NR;
  m.W := nil;
  SetLength(m.W, m.Panels * m.Ktot * KGEMM_NR); // zeros: the extra outputs
  m.B := nil;
  SetLength(m.B, m.Panels * KGEMM_NR);
  for p := 0 to m.Panels - 1 do
    for j := 0 to KGEMM_NR - 1 do
    begin
      o := p * KGEMM_NR + j;
      if o >= aOut then
        break;
      // [panel][tap][input][16] <- w[o][c][k]
      dst := @m.W[p * m.Ktot * KGEMM_NR + j];
      for k := 0 to aTaps - 1 do
        for c := 0 to aCin - 1 do
          dst^[(k * aCin + c) * KGEMM_NR] := w^[(o * aCin + c) * aTaps + k];
      if b <> nil then
        m.B[o] := b^[o];
    end;
end;

procedure KokoroGemmPanel(x: pointer; aXStride, aTapStride: PtrInt;
  const m: TKPanels; aPanel: PtrInt; y: pointer; aYStride, aRows: PtrInt;
  aFlags: PtrInt);
var
  args: TKGemmArgs;
  tile: array[0..KGEMM_MR * KGEMM_NR - 1] of single;
  cols, n, i: PtrInt;
  yp: PtrInt;
begin
  args.A[1] := aXStride;
  args.A[2] := PtrInt(@m.W[aPanel * m.Ktot * KGEMM_NR]);
  args.A[3] := m.Cin;
  args.A[4] := m.Taps;
  args.A[5] := aTapStride;
  args.A[8] := PtrInt(@m.B[aPanel * KGEMM_NR]);
  args.A[9] := aFlags;
  args.A[10] := KGEMM_KB;
  cols := m.OutCount - aPanel * KGEMM_NR;
  if cols > KGEMM_NR then
    cols := KGEMM_NR;
  while aRows > 0 do
  begin
    n := aRows;
    if n > KGEMM_MR then
      n := KGEMM_MR;
    if cols = KGEMM_NR then
    begin
      args.A[6] := PtrInt(y);
      args.A[7] := aYStride;
    end
    else
    begin
      // the last panel, incomplete: by one tile, without overflowing y
      args.A[6] := PtrInt(@tile);
      args.A[7] := KGEMM_NR * SizeOf(single);
      if aFlags and KGEMM_ADD <> 0 then
        for i := 0 to n - 1 do
          MoveFast(pointer(PtrInt(y) + i * aYStride)^, tile[i * KGEMM_NR],
            cols * SizeOf(single));
    end;
    if n = KGEMM_MR then
    begin
      args.A[0] := PtrInt(x);
      KGemm6(args);
    end
    else
    begin
      yp := args.A[6];
      for i := 0 to n - 1 do
      begin
        args.A[0] := PtrInt(x) + i * aXStride;
        args.A[6] := yp + i * args.A[7];
        KGemm1(args);
      end;
      args.A[6] := yp;
    end;
    if cols < KGEMM_NR then
      for i := 0 to n - 1 do
        MoveFast(tile[i * KGEMM_NR], pointer(PtrInt(y) + i * aYStride)^,
          cols * SizeOf(single));
    x := pointer(PtrInt(x) + n * aXStride);
    y := pointer(PtrInt(y) + n * aYStride);
    dec(aRows, n);
  end;
end;

procedure KokoroGemmPair(x: pointer; aXStride, aTapStride: PtrInt;
  const m: TKPanels; aPanel: PtrInt; y: pointer; aYStride, aRows: PtrInt;
  aFlags: PtrInt);
var
  args: TKGemmArgs;
  full: PtrInt;
begin
  if not Assigned(KGemm6x32) or
     (m.OutCount < (aPanel + 2) * KGEMM_NR) then
  begin
    KokoroGemmPanel(x, aXStride, aTapStride, m, aPanel, y, aYStride, aRows, aFlags);
    KokoroGemmPanel(x, aXStride, aTapStride, m, aPanel + 1,
      pointer(PtrInt(y) + KGEMM_NR * SizeOf(single)), aYStride, aRows, aFlags);
    exit;
  end;
  full := aRows - aRows mod KGEMM_MR;
  args.A[1] := aXStride;
  args.A[2] := PtrInt(@m.W[aPanel * m.Ktot * KGEMM_NR]);
  args.A[3] := m.Cin;
  args.A[4] := m.Taps;
  args.A[5] := aTapStride;
  args.A[7] := aYStride;
  args.A[8] := PtrInt(@m.B[aPanel * KGEMM_NR]);
  args.A[9] := aFlags;
  args.A[10] := KGEMM_KB;
  args.A[11] := m.Ktot * KGEMM_NR * SizeOf(single);
  while full > 0 do
  begin
    args.A[0] := PtrInt(x);
    args.A[6] := PtrInt(y);
    KGemm6x32(args);
    x := pointer(PtrInt(x) + KGEMM_MR * aXStride);
    y := pointer(PtrInt(y) + KGEMM_MR * aYStride);
    dec(full, KGEMM_MR);
    dec(aRows, KGEMM_MR);
  end;
  if aRows = 0 then
    exit;
  // the leftover rows: the same sums, panel by panel
  KokoroGemmPanel(x, aXStride, aTapStride, m, aPanel, y, aYStride, aRows, aFlags);
  KokoroGemmPanel(x, aXStride, aTapStride, m, aPanel + 1,
    pointer(PtrInt(y) + KGEMM_NR * SizeOf(single)), aYStride, aRows, aFlags);
end;

function KokoroGemmPairs: boolean;
begin
  result := Assigned(KGemm6x32);
end;

function KokoroGemmKernel: RawUtf8;
begin
  result := KGemmName;
end;

function KokoroSin(x: single): single;
var
  q, r, z, s, c: single;
  n: integer;
begin
  q := x * SNAKE_CONSTS[0];
  if q >= 0 then
    n := trunc(q + SNAKE_CONSTS[1])
  else
    n := -trunc(SNAKE_CONSTS[1] - q);
  q := n;
  // x = n pi / 2 + r, |r| <= pi / 4; pi / 2 in three pieces, in single
  r := x - q * SNAKE_CONSTS[2];
  r := r - q * SNAKE_CONSTS[3];
  r := r - q * SNAKE_CONSTS[4];
  z := r * r;
  s := r + r * z * (SNAKE_CONSTS[5] + z * (SNAKE_CONSTS[6] + z * SNAKE_CONSTS[7]));
  c := SNAKE_CONSTS[11] - SNAKE_CONSTS[1] * z + z * z * (SNAKE_CONSTS[8] + z *
    (SNAKE_CONSTS[9] + z * SNAKE_CONSTS[10]));
  case n and 3 of
    0:
      result := s;
    1:
      result := c;
    2:
      result := -s;
  else
    result := -c;
  end;
end;

function KSigmoid(x: single): single;
begin
  result := 1 / (1 + Exp(-x));
end;

function KTanh(x: single): single;
var
  e: double;
begin
  if x > 20 then
    result := 1
  else if x < -20 then
    result := -1
  else
  begin
    e := Exp(2 * double(x));
    result := (e - 1) / (e + 1);
  end;
end;

procedure KokoroLstmCell(gx, gh, b, c, h: PSingleArray; aHidden: PtrInt);
var
  args: TKGemmArgs;
  j, j0: PtrInt;
  i, f, g, o: single;
begin
  j0 := 0;
  if KHasAvx2 then
  begin
    j0 := aHidden and not 3;
    args.A[0] := PtrInt(gx);
    args.A[1] := PtrInt(gh);
    args.A[2] := PtrInt(b);
    args.A[3] := PtrInt(c);
    args.A[4] := PtrInt(h);
    args.A[5] := aHidden;
    args.A[6] := PtrInt(@KLstmConsts);
    {$ifdef KGEMMAVX2}
    KLstmCellAvx2(args);
    {$endif KGEMMAVX2}
  end;
  for j := j0 to aHidden - 1 do
  begin
    i := KSigmoid(gx^[j] + gh^[j] + b^[j]);
    f := KSigmoid(gx^[aHidden + j] + gh^[aHidden + j] + b^[aHidden + j]);
    g := KTanh(gx^[2 * aHidden + j] + gh^[2 * aHidden + j] + b^[2 * aHidden + j]);
    o := KSigmoid(gx^[3 * aHidden + j] + gh^[3 * aHidden + j] + b^[3 * aHidden + j]);
    c^[j] := f * c^[j] + i * g;
    h^[j] := o * KTanh(c^[j]);
  end;
end;

procedure KokoroColSums(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  aPart: PDoubleArray);
var
  args: TKGemmArgs;
  r, c, c0: PtrInt;
  row: PSingleArray;
begin
  c0 := 0;
  if KHasAvx2 then
  begin
    c0 := aCols and not 15;
    args.A[0] := PtrInt(x);
    args.A[1] := aRows;
    args.A[2] := c0;
    args.A[3] := aStride * SizeOf(single);
    args.A[4] := PtrInt(aPart);
    {$ifdef KGEMMAVX2}
    KColSumsAvx2(args);
    {$endif KGEMMAVX2}
  end;
  for r := 0 to aRows - 1 do
  begin
    row := @x^[r * aStride];
    for c := c0 to aCols - 1 do
      aPart^[c] := aPart^[c] + row^[c];
  end;
end;

procedure KokoroColSquares(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  aMean, aPart: PDoubleArray);
var
  args: TKGemmArgs;
  r, c, c0: PtrInt;
  row: PSingleArray;
  d: double;
begin
  c0 := 0;
  if KHasAvx2 then
  begin
    c0 := aCols and not 15;
    args.A[0] := PtrInt(x);
    args.A[1] := aRows;
    args.A[2] := c0;
    args.A[3] := aStride * SizeOf(single);
    args.A[4] := PtrInt(aPart);
    args.A[5] := PtrInt(aMean);
    {$ifdef KGEMMAVX2}
    KColSquaresAvx2(args);
    {$endif KGEMMAVX2}
  end;
  for r := 0 to aRows - 1 do
  begin
    row := @x^[r * aStride];
    for c := c0 to aCols - 1 do
    begin
      d := row^[c] - aMean^[c];
      aPart^[c] := aPart^[c] + d * d;
    end;
  end;
end;

procedure KokoroAffineSnake(x: PSingleArray; aRows, aCols, aXStride: PtrInt;
  y: PSingleArray; aYStride: PtrInt; aScale, aShift, aAlpha, aInv: PSingleArray);
var
  args: TKGemmArgs;
  r, c, c0: PtrInt;
  xr, yr: PSingleArray;
  v, s: single;
begin
  c0 := 0;
  if KHasAvx2 then
  begin
    c0 := aCols and not 7;
    args.A[0] := PtrInt(x);
    args.A[1] := aRows;
    args.A[2] := c0;
    args.A[3] := aXStride * SizeOf(single);
    args.A[4] := PtrInt(y);
    args.A[5] := aYStride * SizeOf(single);
    args.A[6] := PtrInt(aScale);
    args.A[7] := PtrInt(aShift);
    args.A[8] := PtrInt(aAlpha);
    args.A[9] := PtrInt(aInv);
    args.A[10] := PtrInt(@KSnakeConsts);
    {$ifdef KGEMMAVX2}
    KSnakeAvx2(args);
    {$endif KGEMMAVX2}
  end;
  if c0 = aCols then
    exit;
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aXStride];
    yr := @y^[r * aYStride];
    for c := c0 to aCols - 1 do
    begin
      v := xr^[c];
      if aScale <> nil then
        v := v * aScale^[c] + aShift^[c];
      if aAlpha <> nil then
      begin
        // Snake1D: x + (1 / a) * sin(a * x)^2, per channel
        s := KokoroSin(aAlpha^[c] * v);
        v := v + aInv^[c] * (s * s);
      end;
      yr^[c] := v;
    end;
  end;
end;

procedure InitKernels;
var
  i, j: PtrInt;
  f: double;
begin
  // the constants of KLstmCellAvx2, each on 4 lanes
  for j := 0 to 3 do
  begin
    KLstmConsts[0][j] := 1.4426950408889634;      // 1 / ln 2
    PInt64(@KLstmConsts[1][j])^ := $3fe62e42fee00000; // ln 2, high 32 bits
    PInt64(@KLstmConsts[2][j])^ := $3dea39ef35793c76; // ln 2, the rest
    f := 1;
    for i := 0 to 13 do
    begin
      KLstmConsts[3 + i][j] := 1 / f; // 1 / i!
      f := f * (i + 1);
    end;
    KLstmConsts[17][j] := 1;
    PInt64(@KLstmConsts[18][j])^ := Int64($8000000000000000);
    PInt64(@KLstmConsts[19][j])^ := 1023;
    KLstmConsts[20][j] := 700;
    KLstmConsts[21][j] := -700;
    KLstmConsts[22][j] := 20;
    KLstmConsts[23][j] := -20;
  end;
  // the constants of KSnakeAvx2, each on 8 lanes
  for i := 0 to high(SNAKE_CONSTS) do
    for j := 0 to 7 do
      KSnakeConsts[i][j] := SNAKE_CONSTS[i];
  for j := 0 to 7 do
  begin
    PCardinal(@KSnakeConsts[12][j])^ := $7fffffff;
    PCardinal(@KSnakeConsts[13][j])^ := $80000000;
    PCardinal(@KSnakeConsts[14][j])^ := 1;
    PCardinal(@KSnakeConsts[15][j])^ := 2;
  end;
  KGemm6 := KGemm6Pascal;
  KGemm1 := KGemm1Pascal;
  KGemm6x32 := nil;
  KGemmName := 'pascal';
  {$ifdef KGEMMAVX2}
  if (cfAVX2 in CpuFeatures) and (cfFMA in CpuFeatures) then
  begin
    KGemm6 := KGemm6Avx2;
    KGemm1 := KGemm1Avx2;
    KGemmName := 'avx2';
    KHasAvx2 := true;
    // AVX-512: the processor and the OS (the zmm registers saved, XCR0)
    if cfAVX512F in IntelAvx512 then
    begin
      KGemm6x32 := KGemm6x32Avx512;
      KGemmName := 'avx512';
    end;
  end;
  {$endif KGEMMAVX2}
end;

procedure KokoroForcePascal(aPascal: boolean);
begin
  InitKernels;
  if not aPascal then
    exit;
  KGemm6 := KGemm6Pascal;
  KGemm1 := KGemm1Pascal;
  KGemm6x32 := nil;
  KGemmName := 'pascal';
  KHasAvx2 := false;
end;

initialization
  InitKernels;

end.
