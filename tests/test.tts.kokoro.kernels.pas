/// Kokoro TTS tests - the CPU kernels against their Pascal loops and a double computation
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - mormot.ai.tts.kokoro.kernels: the product (incomplete rows and panels,
// taps, residual), the per-channel sums of AdaIN and Snake, bit for bit
// - without AVX2, the Pascal loop is compared with itself
unit test.tts.kokoro.kernels;

{$I mormot.defines.inc}

{$ifndef FPC}
  {$EXCESSPRECISION OFF}
{$endif FPC}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.tts.kokoro.kernels;

type
  TTestKokoroKernels = class(TSynTestCase)
  protected
    fSeed: cardinal;
    function Rnd: single;
    procedure Fill(var a: TSingleDynArray; aCount: PtrInt; aLow, aHigh: single);
  published
    procedure GemmMatchesDouble;
    procedure ColumnSumsMatchPascal;
    procedure SnakeMatchesPascal;
    procedure SinIsAccurate;
    procedure LstmCellMatchesPascal;
  end;


implementation

function TTestKokoroKernels.Rnd: single;
begin
  // xorshift32: the same data at each run
  fSeed := fSeed xor (fSeed shl 13);
  fSeed := fSeed xor (fSeed shr 17);
  fSeed := fSeed xor (fSeed shl 5);
  result := (fSeed shr 8) / 16777216;
end;

procedure TTestKokoroKernels.Fill(var a: TSingleDynArray; aCount: PtrInt;
  aLow, aHigh: single);
var
  i: PtrInt;
begin
  SetLength(a, aCount);
  for i := 0 to aCount - 1 do
    a[i] := aLow + (aHigh - aLow) * Rnd;
end;

procedure TTestKokoroKernels.GemmMatchesDouble;
const
  // rows (an incomplete tile), inputs, taps and dilation, outputs (an
  // incomplete panel)
  ROWS = 23;
  CIN = 37;
  TAPS = 3;
  DIL = 2;
  OUTS = 40;
var
  m: TKPanels;
  w, b, x, y, y0, y2: TSingleDynArray;
  ref: array of double;
  pascal, add: boolean;
  r, o, k, c, p, xrows: PtrInt;
  acc, d, worst: double;
begin
  fSeed := 12345;
  Fill(w, OUTS * CIN * TAPS, -1, 1);
  Fill(b, OUTS, -1, 1);
  xrows := ROWS + (TAPS - 1) * DIL;
  Fill(x, xrows * CIN, -1, 1);
  Fill(y0, ROWS * OUTS, -1, 1);
  KokoroPack(m, pointer(w), OUTS, CIN, TAPS, pointer(b));
  CheckEqual(m.Panels, 3, 'panels');
  SetLength(ref, ROWS * OUTS);
  for r := 0 to ROWS - 1 do
    for o := 0 to OUTS - 1 do
    begin
      acc := b[o];
      for k := 0 to TAPS - 1 do
        for c := 0 to CIN - 1 do
          acc := acc + double(x[(r + k * DIL) * CIN + c]) * w[(o * CIN + c) * TAPS + k];
      ref[r * OUTS + o] := acc;
    end;
  for pascal := false to true do
  begin
    KokoroForcePascal(pascal);
    try
      for add := false to true do
      begin
        y := copy(y0);
        for p := 0 to m.Panels - 1 do
          KokoroGemmPanel(pointer(x), CIN * SizeOf(single), DIL * CIN * SizeOf(single),
            m, p, @y[p * KGEMM_NR], OUTS * SizeOf(single), ROWS, ord(add) * KGEMM_ADD);
        worst := 0;
        for r := 0 to ROWS * OUTS - 1 do
        begin
          d := ref[r];
          if add then
            d := d + y0[r];
          d := abs(y[r] - d);
          if d > worst then
            worst := d;
        end;
        CheckUtf8(worst < 2e-5, '% add=%: max |diff| %',
          [KokoroGemmKernel, ord(add), worst]);
        // two panels at a time (AVX-512), then the last one: the same bits
        y2 := copy(y0);
        KokoroGemmPair(pointer(x), CIN * SizeOf(single), DIL * CIN * SizeOf(single),
          m, 0, @y2[0], OUTS * SizeOf(single), ROWS, ord(add) * KGEMM_ADD);
        KokoroGemmPanel(pointer(x), CIN * SizeOf(single), DIL * CIN * SizeOf(single),
          m, 2, @y2[2 * KGEMM_NR], OUTS * SizeOf(single), ROWS, ord(add) * KGEMM_ADD);
        CheckUtf8(CompareMem(pointer(y), pointer(y2), ROWS * OUTS * SizeOf(single)),
          '% add=%: pair', [KokoroGemmKernel, ord(add)]);
      end;
    finally
      KokoroForcePascal(false);
    end;
  end;
end;

procedure TTestKokoroKernels.ColumnSumsMatchPascal;
const
  ROWS = 300;
  COLS = 70;
var
  x: TSingleDynArray;
  mean, s1, s2, q1, q2: array of double;
  c: PtrInt;
begin
  fSeed := 777;
  Fill(x, ROWS * COLS, -3, 5);
  SetLength(mean, COLS);
  for c := 0 to COLS - 1 do
    mean[c] := Rnd;
  SetLength(s1, COLS);
  SetLength(s2, COLS);
  SetLength(q1, COLS);
  SetLength(q2, COLS);
  // extra columns (aStride > aCols) and starting totals
  for c := 0 to COLS - 1 do
  begin
    s1[c] := c;
    s2[c] := c;
  end;
  KokoroColSums(pointer(x), ROWS, COLS - 3, COLS, pointer(s1));
  KokoroColSquares(pointer(x), ROWS, COLS - 3, COLS, pointer(mean), pointer(q1));
  KokoroForcePascal(true);
  try
    KokoroColSums(pointer(x), ROWS, COLS - 3, COLS, pointer(s2));
    KokoroColSquares(pointer(x), ROWS, COLS - 3, COLS, pointer(mean), pointer(q2));
  finally
    KokoroForcePascal(false);
  end;
  Check(CompareMem(pointer(s1), pointer(s2), COLS * SizeOf(double)), 'sums');
  Check(CompareMem(pointer(q1), pointer(q2), COLS * SizeOf(double)), 'squares');
  Check(s1[COLS - 1] = COLS - 1, 'untouched column');
end;

procedure TTestKokoroKernels.SnakeMatchesPascal;
const
  ROWS = 37;
  COLS = 132; // 16 x 8 + 4: the end of the row in Pascal
var
  x, scale, shift, alpha, inv, y1, y2: TSingleDynArray;
  mode, c: PtrInt;
  sc, sh, al: PSingleArray;
begin
  fSeed := 4242;
  Fill(x, ROWS * COLS, -60, 60);
  Fill(scale, COLS, -3, 3);
  Fill(shift, COLS, -2, 2);
  Fill(alpha, COLS, 0.05, 8);
  SetLength(inv, COLS);
  for c := 0 to COLS - 1 do
    inv[c] := 1 / alpha[c];
  for mode := 0 to 2 do
  begin
    // AdaIN then Snake, AdaIN alone, Snake alone
    sc := pointer(scale);
    sh := pointer(shift);
    al := pointer(alpha);
    if mode = 1 then
      al := nil
    else if mode = 2 then
    begin
      sc := nil;
      sh := nil;
    end;
    SetLength(y1, ROWS * COLS);
    SetLength(y2, ROWS * COLS);
    KokoroAffineSnake(pointer(x), ROWS, COLS, COLS, pointer(y1), COLS, sc, sh, al,
      pointer(inv));
    KokoroForcePascal(true);
    try
      KokoroAffineSnake(pointer(x), ROWS, COLS, COLS, pointer(y2), COLS, sc, sh, al,
        pointer(inv));
    finally
      KokoroForcePascal(false);
    end;
    CheckUtf8(CompareMem(pointer(y1), pointer(y2), ROWS * COLS * SizeOf(single)),
      'snake mode %', [mode]);
  end;
  // in place
  y1 := copy(x);
  KokoroAffineSnake(pointer(y1), ROWS, COLS, COLS, pointer(y1), COLS, pointer(scale),
    pointer(shift), pointer(alpha), pointer(inv));
  KokoroAffineSnake(pointer(x), ROWS, COLS, COLS, pointer(y2), COLS, pointer(scale),
    pointer(shift), pointer(alpha), pointer(inv));
  Check(CompareMem(pointer(y1), pointer(y2), ROWS * COLS * SizeOf(single)), 'in place');
end;

procedure TTestKokoroKernels.SinIsAccurate;
var
  i: PtrInt;
  v: single;
  d, worst: double;
begin
  fSeed := 99;
  worst := 0;
  for i := 1 to 100000 do
  begin
    v := (Rnd - 0.5) * 2000;
    d := abs(KokoroSin(v) - Sin(double(v)));
    if d > worst then
      worst := d;
  end;
  CheckUtf8(worst < 1e-6, 'sin max |diff| %', [worst]);
end;

procedure TTestKokoroKernels.LstmCellMatchesPascal;
const
  H = 258; // 64 x 4 + 2: the end in Pascal
var
  gx, gh, b, c0, c1, c2, h1, h2: TSingleDynArray;
  i, same: PtrInt;
begin
  fSeed := 31337;
  Fill(gx, 4 * H, -12, 12);
  Fill(gh, 4 * H, -6, 6);
  Fill(b, 4 * H, -1, 1);
  Fill(c0, H, -25, 25);
  // the bounds: saturated sigmoids, tangents beyond +-20
  gx[0] := 900;
  gx[1] := -705; // Exp(705): no overflow in Pascal
  gh[1] := 0;
  b[1] := 0;
  gx[2 * H + 2] := 30;
  gx[2 * H + 3] := -30;
  c0[4] := 40;
  c0[5] := -40;
  SetLength(h1, H);
  SetLength(h2, H);
  c1 := copy(c0);
  c2 := copy(c0);
  KokoroLstmCell(pointer(gx), pointer(gh), pointer(b), pointer(c1), pointer(h1), H);
  KokoroForcePascal(true);
  try
    KokoroLstmCell(pointer(gx), pointer(gh), pointer(b), pointer(c2), pointer(h2), H);
  finally
    KokoroForcePascal(false);
  end;
  same := 0;
  for i := 0 to H - 1 do
    if (PCardinal(@c1[i])^ = PCardinal(@c2[i])^) and
       (PCardinal(@h1[i])^ = PCardinal(@h2[i])^) then
      inc(same);
  CheckUtf8(same = H, 'lstm cell: % of % units identical', [same, H]);
end;

end.
