/// Kokoro TTS tests - text to Kokoro phonemes through espeak-ng, against Python
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - against misaki and phonemizer, in tools/tts/kokoro_ref.py
// - without espeak-ng or references (see test.tts.kokoro.ref), skipped
unit test.tts.kokoro.g2p;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.test,
  mormot.ai.tts.espeak,
  test.tts.kokoro.ref;

type
  TTestKokoroG2P = class(TSynTestCase)
  protected
    fEspeak: TTtsEspeak;
    fMissing: boolean;
    function Espeak: TTtsEspeak;
  public
    destructor Destroy; override;
  published
    procedure RawEspeakMatchesPhonemizer;
    procedure PhonemesMatchMisaki;
  end;


implementation

destructor TTestKokoroG2P.Destroy;
begin
  fEspeak.Free;
  inherited Destroy;
end;

function TTestKokoroG2P.Espeak: TTtsEspeak;
var
  folder: TFileName;
begin
  if (fEspeak = nil) and not fMissing then
  begin
    folder := KokoroTestEspeak;
    if folder = '' then
    begin
      fMissing := true;
      AddConsole(KokoroNoEspeakText);
    end
    else
      fEspeak := TTtsEspeak.Create(folder, 'fr-fr');
  end;
  result := fEspeak;
end;

procedure TTestKokoroG2P.RawEspeakMatchesPhonemizer;
var
  ref: TKokoroRef;
  r: PtrInt;
  u: UnicodeString;
  mine, want: RawUtf8;
begin
  if Espeak = nil then
    exit;
  for r := 0 to high(KOKORO_REFS) do
  begin
    if not KokoroLoadRef(KOKORO_REFS[r], ref) then
    begin
      AddConsole('skipped: no reference % (tools/tts/kokoro_ref.py)', [KOKORO_REFS[r]]);
      continue;
    end;
    // what misaki gives phonemizer: curly quotes, angle quotes
    u := Utf8ToString(ref.Doc.U['text']);
    u := StringReplace(u, #$00AB, #$201C, [rfReplaceAll]);
    u := StringReplace(u, #$00BB, #$201D, [rfReplaceAll]);
    u := StringReplace(u, '(', #$00AB, [rfReplaceAll]);
    u := StringReplace(u, ')', #$00BB, [rfReplaceAll]);
    mine := Espeak.Phonemize(StringToUtf8(u));
    want := ref.Doc.U['espeak_raw'];
    CheckUtf8(mine = want, '%:'#10'  mine "%"'#10'  want "%"', [KOKORO_REFS[r], mine, want]);
  end;
end;

procedure TTestKokoroG2P.PhonemesMatchMisaki;
var
  ref: TKokoroRef;
  r: PtrInt;
  mine, want: RawUtf8;
begin
  if Espeak = nil then
    exit;
  for r := 0 to high(KOKORO_REFS) do
  begin
    if not KokoroLoadRef(KOKORO_REFS[r], ref) then
      continue;
    mine := Espeak.Kokoro(ref.Doc.U['text']);
    want := ref.Doc.U['phonemes'];
    CheckUtf8(mine = want, '%:'#10'  mine "%"'#10'  want "%"', [KOKORO_REFS[r], mine, want]);
    AddConsole('% -> %', [ref.Doc.U['text'], mine], true);
  end;
end;

end.
