/// Text-to-speech - the tensors of a PyTorch file (.pth / .pt from torch.save), without Python
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the format: a zip without compression (archive/data.pkl,
// archive/data/0..N, archive/byteorder); data.pkl is a pickle (protocol 2
// to 5) describing the saved object - a state_dict dictionary for Kokoro, a
// tensor for a voice - and naming each storage by its name in the zip
// (BINPERSID)
// - only the opcodes and objects torch.save uses for tensors are read:
// dictionaries, OrderedDict, tuples, lists, numbers, strings,
// torch._utils._rebuild_tensor_v2 and torch.*Storage; any other object
// raises an exception
// - the file is loaded in memory once: the contiguous float32 tensors point
// right into it
unit mormot.ai.tts.torchfile;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text;

type
  ETorchFile = class(ESynException);

  TTorchDType = (
    tdUnknown,
    tdF32,
    tdF16,
    tdBF16,
    tdF64,
    tdI64,
    tdI32,
    tdU8);

  /// a tensor of the file
  TTorchTensor = record
    /// dotted name: 'decoder.module.decode.0.conv1.weight_v'; '' for a
    // file holding a single tensor
    Name: RawUtf8;
    DType: TTorchDType;
    Shape: TInt64DynArray;
    /// the number of elements
    Count: Int64;
    /// the bytes of the first element, in the loaded file
    Data: pointer;
  end;
  PTorchTensor = ^TTorchTensor;

  TTorchFile = class
  protected
    fFileName: TFileName;
    fContent: RawByteString;
    fNames: TRawUtf8DynArray;   // the zip entries
    fStart, fSize: TInt64DynArray;
    fTensors: array of TTorchTensor;
    function Entry(const aName: RawUtf8; out aData: PAnsiChar): Int64;
    procedure ReadZip;
    procedure ReadPickle(const aPrefix: RawUtf8);
  public
    /// loads and decodes aFileName; raises ETorchFile if it is not a
    // readable torch.save file
    constructor Create(const aFileName: TFileName);
    /// the index of aName, -1 if missing
    function IndexOf(const aName: RawUtf8): PtrInt;
    /// the tensor aName; raises ETorchFile if missing
    function Tensor(const aName: RawUtf8): PTorchTensor;
    /// the float32 values of aName, of shape aDims; raises ETorchFile if the
    // shape or the type differs
    function F32(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
    function Count: PtrInt;
    function Item(aIndex: PtrInt): PTorchTensor;
    property FileName: TFileName
      read fFileName;
  end;

function TorchShapeText(const aShape: TInt64DynArray): RawUtf8;


implementation

function TorchShapeText(const aShape: TInt64DynArray): RawUtf8;
var
  i: PtrInt;
begin
  result := '[';
  for i := 0 to high(aShape) do
  begin
    if i > 0 then
      result := result + ',';
    result := result + Int64ToUtf8(aShape[i]);
  end;
  result := result + ']';
end;


{ the pickle }

type
  TPickleKind = (
    pkNone,
    pkBool,
    pkInt,
    pkFloat,
    pkStr,
    pkTuple,
    pkList,
    pkDict,
    pkGlobal,
    pkStorage,
    pkTensor,
    pkMark);

  TPickle = class
  public
    Kind: TPickleKind;
    Int: Int64;
    Float: double;
    Str: RawUtf8;         // string; module.name of a global; key of a storage
    Items: array of TPickle;   // tuple, list, values of a dictionary
    Keys: array of TPickle;    // keys of a dictionary
    DType: TTorchDType;   // storage, tensor
    Offset: Int64;        // tensor: first element in the storage
    Shape, Stride: TInt64DynArray;
    procedure Add(aItem: TPickle);
    procedure SetItem(aKey, aValue: TPickle);
  end;

procedure TPickle.Add(aItem: TPickle);
begin
  SetLength(Items, length(Items) + 1);
  Items[high(Items)] := aItem;
end;

procedure TPickle.SetItem(aKey, aValue: TPickle);
begin
  SetLength(Keys, length(Keys) + 1);
  Keys[high(Keys)] := aKey;
  Add(aValue);
end;

function StorageType(const aName: RawUtf8): TTorchDType;
begin
  if aName = 'torch.FloatStorage' then
    result := tdF32
  else if aName = 'torch.HalfStorage' then
    result := tdF16
  else if aName = 'torch.BFloat16Storage' then
    result := tdBF16
  else if aName = 'torch.DoubleStorage' then
    result := tdF64
  else if aName = 'torch.LongStorage' then
    result := tdI64
  else if aName = 'torch.IntStorage' then
    result := tdI32
  else if aName = 'torch.ByteStorage' then
    result := tdU8
  else
    result := tdUnknown;
end;

const
  DTYPE_SIZE: array[TTorchDType] of integer = (0, 4, 2, 2, 8, 8, 4, 1);


{ TTorchFile }

constructor TTorchFile.Create(const aFileName: TFileName);
var
  i: PtrInt;
  prefix: RawUtf8;
  data: PAnsiChar;
begin
  inherited Create;
  fFileName := aFileName;
  fContent := StringFromFile(aFileName);
  if length(fContent) < 22 then
    raise ETorchFile.CreateUtf8('%: cannot read %', [self, aFileName]);
  ReadZip;
  prefix := '';
  for i := 0 to high(fNames) do
    if (length(fNames[i]) > 9) and
       (copy(fNames[i], length(fNames[i]) - 8, 9) = '/data.pkl') then
    begin
      prefix := copy(fNames[i], 1, length(fNames[i]) - 9);
      break;
    end;
  if prefix = '' then
    raise ETorchFile.CreateUtf8('%: no data.pkl in %', [self, aFileName]);
  i := Entry(prefix + '/byteorder', data);
  if (i > 0) and not IdemPChar(PUtf8Char(data), 'LITTLE') then
    raise ETorchFile.CreateUtf8('%: % is not little-endian', [self, aFileName]);
  ReadPickle(prefix);
end;

procedure TTorchFile.ReadZip;
var
  p, eocd: PtrInt;
  n, i, at, nameLen, extraLen, commentLen, method, localNameLen,
    localExtraLen: PtrInt;
  b: PByteArray;
  size, offset: Int64;

  function U16(aAt: PtrInt): cardinal;
  begin
    result := PWord(@b^[aAt])^;
  end;

  function U32(aAt: PtrInt): cardinal;
  begin
    result := PCardinal(@b^[aAt])^;
  end;

begin
  b := pointer(fContent);
  // the end of the central directory: PK 05 06, at most 64 KB of comment
  eocd := -1;
  p := length(fContent) - 22;
  while (p >= 0) and (p >= length(fContent) - 22 - 65535) do
  begin
    if U32(p) = $06054b50 then
    begin
      eocd := p;
      break;
    end;
    dec(p);
  end;
  if eocd < 0 then
    raise ETorchFile.CreateUtf8('%: % is not a zip', [self, fFileName]);
  n := U16(eocd + 10);
  at := U32(eocd + 16);
  if (n = $FFFF) or (cardinal(at) = $FFFFFFFF) then
    raise ETorchFile.CreateUtf8('%: zip64 not supported (%)', [self, fFileName]);
  SetLength(fNames, n);
  SetLength(fStart, n);
  SetLength(fSize, n);
  for i := 0 to n - 1 do
  begin
    if (at + 46 > length(fContent)) or (U32(at) <> $02014b50) then
      raise ETorchFile.CreateUtf8('%: broken zip directory in %', [self, fFileName]);
    method := U16(at + 10);
    size := U32(at + 20);
    nameLen := U16(at + 28);
    extraLen := U16(at + 30);
    commentLen := U16(at + 32);
    offset := U32(at + 42);
    FastSetString(fNames[i], @b^[at + 46], nameLen);
    if method <> 0 then
      raise ETorchFile.CreateUtf8('%: % is compressed in %', [self, fNames[i], fFileName]);
    if (size = $FFFFFFFF) or (offset = $FFFFFFFF) then
      raise ETorchFile.CreateUtf8('%: zip64 entry % not supported', [self, fNames[i]]);
    // the local header has its own extra field (torch aligns the data)
    if (offset + 30 > length(fContent)) or (U32(offset) <> $04034b50) then
      raise ETorchFile.CreateUtf8('%: broken zip entry %', [self, fNames[i]]);
    localNameLen := U16(offset + 26);
    localExtraLen := U16(offset + 28);
    fStart[i] := offset + 30 + localNameLen + localExtraLen;
    fSize[i] := size;
    if fStart[i] + size > length(fContent) then
      raise ETorchFile.CreateUtf8('%: truncated zip entry %', [self, fNames[i]]);
    inc(at, 46 + nameLen + extraLen + commentLen);
  end;
end;

function TTorchFile.Entry(const aName: RawUtf8; out aData: PAnsiChar): Int64;
var
  i: PtrInt;
begin
  for i := 0 to high(fNames) do
    if fNames[i] = aName then
    begin
      aData := @PAnsiChar(pointer(fContent))[fStart[i]];
      result := fSize[i];
      exit;
    end;
  aData := nil;
  result := -1;
end;

procedure TTorchFile.ReadPickle(const aPrefix: RawUtf8);
var
  pkl: PAnsiChar;
  size, pos: Int64;
  owned: TList;
  stack: array of TPickle;
  table: array of TPickle;
  memoized: Int64;
  top: TPickle;

  function NewValue(aKind: TPickleKind): TPickle;
  begin
    result := TPickle.Create;
    result.Kind := aKind;
    owned.Add(result);
  end;

  procedure Push(v: TPickle);
  begin
    SetLength(stack, length(stack) + 1);
    stack[high(stack)] := v;
  end;

  function Pop: TPickle;
  begin
    if stack = nil then
      raise ETorchFile.CreateUtf8('%: pickle stack underflow in %', [self, fFileName]);
    result := stack[high(stack)];
    SetLength(stack, length(stack) - 1);
  end;

  function Peek: TPickle;
  begin
    if stack = nil then
      raise ETorchFile.CreateUtf8('%: empty pickle stack in %', [self, fFileName]);
    result := stack[high(stack)];
  end;

  /// the values since the last mark, the mark removed
  function PopMark: TPickle;
  var
    i, m: PtrInt;
  begin
    m := high(stack);
    while (m >= 0) and (stack[m].Kind <> pkMark) do
      dec(m);
    if m < 0 then
      raise ETorchFile.CreateUtf8('%: pickle mark missing in %', [self, fFileName]);
    result := NewValue(pkTuple);
    for i := m + 1 to high(stack) do
      result.Add(stack[i]);
    SetLength(stack, m);
  end;

  function Need(n: Int64): PAnsiChar;
  begin
    if pos + n > size then
      raise ETorchFile.CreateUtf8('%: truncated pickle in %', [self, fFileName]);
    result := @pkl[pos];
    inc(pos, n);
  end;

  procedure Memoize(aIndex: Int64);
  begin
    if aIndex >= length(table) then
      SetLength(table, aIndex + 64);
    if table[aIndex] = nil then
      inc(memoized);
    table[aIndex] := Peek;
  end;

  function Recall(aIndex: Int64): TPickle;
  begin
    if (aIndex >= length(table)) or (table[aIndex] = nil) then
      raise ETorchFile.CreateUtf8('%: pickle memo % missing in %', [self, aIndex, fFileName]);
    result := table[aIndex];
  end;

  function Ints(v: TPickle): TInt64DynArray;
  var
    i: PtrInt;
  begin
    SetLength(result, length(v.Items));
    for i := 0 to high(v.Items) do
      result[i] := v.Items[i].Int;
  end;

  function Reduce(aCallable, aArgs: TPickle): TPickle;
  var
    a: TPickle;
  begin
    if aCallable.Kind <> pkGlobal then
      raise ETorchFile.CreateUtf8('%: REDUCE on a % in %', [self, ord(aCallable.Kind), fFileName]);
    if aCallable.Str = 'collections.OrderedDict' then
      result := NewValue(pkDict)
    else if (aCallable.Str = 'torch._utils._rebuild_tensor_v2') or
            (aCallable.Str = 'torch._utils._rebuild_tensor') then
    begin
      // (storage, storage_offset, size, stride, ...)
      if (length(aArgs.Items) < 4) or (aArgs.Items[0].Kind <> pkStorage) then
        raise ETorchFile.CreateUtf8('%: unexpected tensor in %', [self, fFileName]);
      a := aArgs.Items[0];
      result := NewValue(pkTensor);
      result.Str := a.Str;
      result.DType := a.DType;
      result.Offset := aArgs.Items[1].Int;
      result.Shape := Ints(aArgs.Items[2]);
      result.Stride := Ints(aArgs.Items[3]);
    end
    else
      raise ETorchFile.CreateUtf8('%: unsupported object % in %', [self, aCallable.Str, fFileName]);
  end;

  procedure Collect(const aName: RawUtf8; v: TPickle);
  var
    i: PtrInt;
    t: PTorchTensor;
    data: PAnsiChar;
    bytes, expect: Int64;
    n: RawUtf8;
  begin
    case v.Kind of
      pkDict:
        for i := 0 to high(v.Keys) do
        begin
          if aName = '' then
            n := v.Keys[i].Str
          else
            n := aName + '.' + v.Keys[i].Str;
          Collect(n, v.Items[i]);
        end;
      pkTensor:
        begin
          SetLength(fTensors, length(fTensors) + 1);
          t := @fTensors[high(fTensors)];
          t^.Name := aName;
          t^.DType := v.DType;
          t^.Shape := v.Shape;
          t^.Count := 1;
          for i := 0 to high(v.Shape) do
            t^.Count := t^.Count * v.Shape[i];
          // contiguous: each stride is the product of the following dimensions
          expect := 1;
          for i := high(v.Shape) downto 0 do
          begin
            if (v.Shape[i] > 1) and (v.Stride[i] <> expect) then
              raise ETorchFile.CreateUtf8('%: % is not contiguous', [self, aName]);
            expect := expect * v.Shape[i];
          end;
          bytes := Entry(aPrefix + '/data/' + v.Str, data);
          if (bytes < 0) or (DTYPE_SIZE[v.DType] = 0) or
             ((v.Offset + t^.Count) * DTYPE_SIZE[v.DType] > bytes) then
            raise ETorchFile.CreateUtf8('%: storage % of % missing or too short',
              [self, v.Str, aName]);
          t^.Data := data + v.Offset * DTYPE_SIZE[v.DType];
        end;
    end;
  end;

var
  op: AnsiChar;
  v, w, k: TPickle;
  i: PtrInt;
  n: Int64;
  s: RawUtf8;
  p: PAnsiChar;
begin
  size := Entry(aPrefix + '/data.pkl', pkl);
  pos := 0;
  memoized := 0;
  top := nil;
  owned := TList.Create;
  try
    repeat
      op := Need(1)^;
      case op of
        #$80: // PROTO
          Need(1);
        #$95: // FRAME
          Need(8);
        '.': // STOP
          begin
            top := Pop;
            break;
          end;
        '(': // MARK
          Push(NewValue(pkMark));
        '}': // EMPTY_DICT
          Push(NewValue(pkDict));
        ']': // EMPTY_LIST
          Push(NewValue(pkList));
        ')': // EMPTY_TUPLE
          Push(NewValue(pkTuple));
        'N':
          Push(NewValue(pkNone));
        #$88, #$89: // NEWTRUE, NEWFALSE
          begin
            v := NewValue(pkBool);
            v.Int := ord(op = #$88);
            Push(v);
          end;
        'K': // BININT1
          begin
            v := NewValue(pkInt);
            v.Int := PByte(Need(1))^;
            Push(v);
          end;
        'M': // BININT2
          begin
            v := NewValue(pkInt);
            v.Int := PWord(Need(2))^;
            Push(v);
          end;
        'J': // BININT
          begin
            v := NewValue(pkInt);
            v.Int := PInteger(Need(4))^;
            Push(v);
          end;
        #$8a: // LONG1: signed little-endian
          begin
            n := PByte(Need(1))^;
            p := Need(n);
            v := NewValue(pkInt);
            v.Int := 0;
            for i := n - 1 downto 0 do
              v.Int := (v.Int shl 8) or PByteArray(p)^[i];
            if (n > 0) and (n < 8) and (PByteArray(p)^[n - 1] and $80 <> 0) then
              v.Int := v.Int - (Int64(1) shl (8 * n));
            Push(v);
          end;
        'G': // BINFLOAT: big-endian
          begin
            p := Need(8);
            v := NewValue(pkFloat);
            for i := 0 to 7 do
              PByteArray(@v.Float)^[i] := PByteArray(p)^[7 - i];
            Push(v);
          end;
        'X', #$8c, #$8d: // BINUNICODE, SHORT_BINUNICODE, BINUNICODE8
          begin
            if op = 'X' then
              n := PCardinal(Need(4))^
            else if op = #$8c then
              n := PByte(Need(1))^
            else
              n := PInt64(Need(8))^;
            v := NewValue(pkStr);
            FastSetString(v.Str, Need(n), n);
            Push(v);
          end;
        'c': // GLOBAL: 'module\nname\n'
          begin
            s := '';
            repeat
              p := Need(1);
              if p^ = #10 then
                break;
              AppendBufferToUtf8(PUtf8Char(p), 1, s);
            until false;
            s := s + '.';
            repeat
              p := Need(1);
              if p^ = #10 then
                break;
              AppendBufferToUtf8(PUtf8Char(p), 1, s);
            until false;
            v := NewValue(pkGlobal);
            v.Str := s;
            Push(v);
          end;
        #$93: // STACK_GLOBAL
          begin
            w := Pop;
            k := Pop;
            v := NewValue(pkGlobal);
            v.Str := k.Str + '.' + w.Str;
            Push(v);
          end;
        'q': // BINPUT
          Memoize(PByte(Need(1))^);
        'r': // LONG_BINPUT
          Memoize(PCardinal(Need(4))^);
        #$94: // MEMOIZE
          Memoize(memoized);
        'h': // BINGET
          Push(Recall(PByte(Need(1))^));
        'j': // LONG_BINGET
          Push(Recall(PCardinal(Need(4))^));
        't': // TUPLE
          Push(PopMark);
        #$85, #$86, #$87: // TUPLE1, TUPLE2, TUPLE3
          begin
            v := NewValue(pkTuple);
            n := ord(op) - $84;
            SetLength(v.Items, n);
            for i := n - 1 downto 0 do
              v.Items[i] := Pop;
            Push(v);
          end;
        'Q': // BINPERSID: ('storage', type, key, location, count)
          begin
            w := Pop;
            if (w.Kind <> pkTuple) or (length(w.Items) < 3) or
               (w.Items[0].Str <> 'storage') then
              raise ETorchFile.CreateUtf8('%: unexpected persistent id in %', [self, fFileName]);
            v := NewValue(pkStorage);
            v.DType := StorageType(w.Items[1].Str);
            v.Str := w.Items[2].Str;
            if v.DType = tdUnknown then
              raise ETorchFile.CreateUtf8('%: storage type % not supported', [self, w.Items[1].Str]);
            Push(v);
          end;
        'R': // REDUCE
          begin
            w := Pop;
            v := Pop;
            Push(Reduce(v, w));
          end;
        'b': // BUILD: the state (the _metadata of a state_dict) is ignored
          Pop;
        's': // SETITEM
          begin
            w := Pop;
            k := Pop;
            Peek.SetItem(k, w);
          end;
        'u': // SETITEMS
          begin
            w := PopMark;
            v := Peek;
            i := 0;
            while i + 1 <= high(w.Items) do
            begin
              v.SetItem(w.Items[i], w.Items[i + 1]);
              inc(i, 2);
            end;
          end;
        'a': // APPEND
          begin
            w := Pop;
            Peek.Add(w);
          end;
        'e': // APPENDS
          begin
            w := PopMark;
            v := Peek;
            for i := 0 to high(w.Items) do
              v.Add(w.Items[i]);
          end;
      else
        raise ETorchFile.CreateUtf8('%: pickle opcode $% not supported in %',
          [self, ord(op), fFileName]);
      end;
    until false;
    if top = nil then
      raise ETorchFile.CreateUtf8('%: empty pickle in %', [self, fFileName]);
    Collect('', top);
  finally
    for i := 0 to owned.Count - 1 do
      TPickle(owned[i]).Free;
    owned.Free;
  end;
end;

function TTorchFile.IndexOf(const aName: RawUtf8): PtrInt;
begin
  for result := 0 to high(fTensors) do
    if fTensors[result].Name = aName then
      exit;
  result := -1;
end;

function TTorchFile.Tensor(const aName: RawUtf8): PTorchTensor;
var
  i: PtrInt;
begin
  i := IndexOf(aName);
  if i < 0 then
    raise ETorchFile.CreateUtf8('%: no tensor % in %', [self, aName, fFileName]);
  result := @fTensors[i];
end;

function TTorchFile.F32(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
var
  t: PTorchTensor;
  i: PtrInt;
  ok: boolean;
  want: TInt64DynArray;
begin
  t := Tensor(aName);
  ok := (t^.DType = tdF32) and (length(t^.Shape) = length(aDims));
  if ok then
    for i := 0 to high(aDims) do
      if t^.Shape[i] <> aDims[i] then
        ok := false;
  if not ok then
  begin
    SetLength(want, length(aDims));
    for i := 0 to high(aDims) do
      want[i] := aDims[i];
    raise ETorchFile.CreateUtf8('%: % is % %, expected float32 %', [self, aName,
      ord(t^.DType), TorchShapeText(t^.Shape), TorchShapeText(want)]);
  end;
  result := t^.Data;
end;

function TTorchFile.Count: PtrInt;
begin
  result := length(fTensors);
end;

function TTorchFile.Item(aIndex: PtrInt): PTorchTensor;
begin
  result := @fTensors[aIndex];
end;

end.
