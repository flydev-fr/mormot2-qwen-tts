/// Text-to-speech - PCM helpers: float to 16-bit, WAV files, streamed playback
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - TTtsPcmStream plays PCM while it is computed, through waveOut: unlike
// PlaySound, waveOut tells when a buffer is done (an event) and waveOutReset
// cuts the sound at once - Windows only
unit mormot.ai.tts.wave;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  {$ifdef OSWINDOWS}
  Windows,
  MMSystem,
  {$endif OSWINDOWS}
  mormot.core.base,
  mormot.core.os,
  mormot.core.text;

type
  ETtsAudio = class(ESynException);

  /// returns true when the running sentence must be cut; called at least
  // every 100 ms while speaking
  TOnTtsAbort = function: boolean of object;

/// float samples (-1..1) to signed 16-bit PCM
procedure TtsFloatToPcm16(aSrc: PSingleArray; aCount: PtrInt; aDest: PSmallIntArray);

/// a 16-bit PCM RIFF/WAVE file in memory, from float samples (-1..1)
function TtsWav16(aSamples: PSingleArray; aCount: PtrInt; aSampleRate: cardinal;
  aChannels: word = 1): RawByteString;

{$ifdef OSWINDOWS}

type
  TTtsWaveHeaders = array of PWaveHdr;

  /// plays 16-bit PCM which arrives by pieces, while it is computed: the
  // output opens once, each Write adds a buffer to its queue, Drain waits for
  // the end, Stop cuts at once
  // - to be used from a single thread
  TTtsPcmStream = class
  private
    fOut: HWAVEOUT;
    fDone: THandle;
    fQueue: TTtsWaveHeaders;
    fHead, fCount: PtrInt;
    fBytesPerMs: cardinal;
    fWritten: Int64;
    procedure Reclaim;
  public
    /// raise ETtsAudio if the audio output refuses the format
    constructor Create(aSampleRate: cardinal; aChannels: word = 1);
    /// cut what remains and close the output
    destructor Destroy; override;
    /// add aBytes bytes to play, copied: the output plays them as soon as it
    // has finished the previous ones
    procedure Write(aData: pointer; aBytes: cardinal);
    /// wait for the end of all that was written, or cut as soon as aAbort
    // returns true (checked every 100 ms at most); false if cut
    function Drain(const aAbort: TOnTtsAbort): boolean;
    /// cut now and forget the queue
    procedure Stop;
    /// the milliseconds written and not yet played, within one buffer
    function PendingMs: cardinal;
    /// the bytes written since the creation
    property Written: Int64
      read fWritten;
  end;

{$endif OSWINDOWS}


implementation

procedure TtsFloatToPcm16(aSrc: PSingleArray; aCount: PtrInt; aDest: PSmallIntArray);
var
  i, v: PtrInt;
begin
  for i := 0 to aCount - 1 do
  begin
    v := round(aSrc[i] * 32767);
    if v > 32767 then
      v := 32767
    else if v < -32768 then
      v := -32768;
    aDest[i] := v;
  end;
end;

type
  TTtsWavHeader = packed record
    Riff: cardinal;
    RiffSize: cardinal;
    Wave: cardinal;
    Fmt: cardinal;
    FmtSize: cardinal;
    Format, Channels: word;
    SampleRate, BytesPerSec: cardinal;
    BlockAlign, Bits: word;
    Data: cardinal;
    DataSize: cardinal;
  end;
  PTtsWavHeader = ^TTtsWavHeader;

function TtsWav16(aSamples: PSingleArray; aCount: PtrInt; aSampleRate: cardinal;
  aChannels: word): RawByteString;
var
  h: PTtsWavHeader;
begin
  FastNewRawByteString(result, SizeOf(TTtsWavHeader) + aCount * 2);
  h := pointer(result);
  h^.Riff := $46464952; // 'RIFF'
  h^.RiffSize := length(result) - 8;
  h^.Wave := $45564157; // 'WAVE'
  h^.Fmt := $20746d66;  // 'fmt '
  h^.FmtSize := 16;
  h^.Format := 1;       // PCM
  h^.Channels := aChannels;
  h^.SampleRate := aSampleRate;
  h^.BlockAlign := aChannels * 2;
  h^.BytesPerSec := aSampleRate * h^.BlockAlign;
  h^.Bits := 16;
  h^.Data := $61746164; // 'data'
  h^.DataSize := aCount * 2;
  TtsFloatToPcm16(aSamples, aCount, @PByteArray(result)[SizeOf(TTtsWavHeader)]);
end;


{$ifdef OSWINDOWS}

{ TTtsPcmStream }

constructor TTtsPcmStream.Create(aSampleRate: cardinal; aChannels: word);
var
  f: TWaveFormatEx;
  res: MMRESULT;
begin
  inherited Create;
  fDone := CreateEvent(nil, false, false, nil);
  if fDone = 0 then
    RaiseLastOSError;
  FillCharFast(f, SizeOf(f), 0);
  f.wFormatTag := WAVE_FORMAT_PCM;
  f.nChannels := aChannels;
  f.nSamplesPerSec := aSampleRate;
  f.wBitsPerSample := 16;
  f.nBlockAlign := aChannels * 2;
  f.nAvgBytesPerSec := aSampleRate * f.nBlockAlign;
  fBytesPerMs := f.nAvgBytesPerSec div 1000;
  if fBytesPerMs = 0 then
    fBytesPerMs := 1;
  res := waveOutOpen(@fOut, WAVE_MAPPER, @f, DWORD_PTR(fDone), 0, CALLBACK_EVENT);
  if res <> MMSYSERR_NOERROR then
  begin
    fOut := 0;
    raise ETtsAudio.CreateUtf8('the audio output refused % Hz PCM (waveOutOpen %)',
      [aSampleRate, res]);
  end;
end;

destructor TTtsPcmStream.Destroy;
begin
  if fOut <> 0 then
  begin
    Stop;
    waveOutClose(fOut);
  end;
  if fDone <> 0 then
    CloseHandle(fDone);
  inherited Destroy;
end;

procedure TTtsPcmStream.Reclaim;
var
  h: PWaveHdr;
begin
  // the played buffers, in the order they were written
  while fCount > 0 do
  begin
    h := fQueue[fHead];
    if (h^.dwFlags and WHDR_DONE) = 0 then
      break;
    waveOutUnprepareHeader(fOut, h, SizeOf(TWaveHdr));
    FreeMem(h);
    fQueue[fHead] := nil;
    fHead := (fHead + 1) mod length(fQueue);
    dec(fCount);
  end;
end;

procedure TTtsPcmStream.Write(aData: pointer; aBytes: cardinal);
var
  h: PWaveHdr;
  res: MMRESULT;
  grown: TTtsWaveHeaders;
  i: PtrInt;
begin
  if aBytes = 0 then
    exit;
  Reclaim;
  if fCount = length(fQueue) then
  begin
    // the circular queue grows, in order
    SetLength(grown, fCount * 2 + 16);
    for i := 0 to fCount - 1 do
      grown[i] := fQueue[(fHead + i) mod length(fQueue)];
    fQueue := grown;
    fHead := 0;
  end;
  // the header and the samples in one block: the address does not move
  GetMem(h, SizeOf(TWaveHdr) + aBytes);
  FillCharFast(h^, SizeOf(TWaveHdr), 0);
  h^.lpData := PAnsiChar(h) + SizeOf(TWaveHdr);
  h^.dwBufferLength := aBytes;
  MoveFast(aData^, h^.lpData^, aBytes);
  res := waveOutPrepareHeader(fOut, h, SizeOf(TWaveHdr));
  if res <> MMSYSERR_NOERROR then
  begin
    FreeMem(h);
    raise ETtsAudio.CreateUtf8('waveOutPrepareHeader %', [res]);
  end;
  res := waveOutWrite(fOut, h, SizeOf(TWaveHdr));
  if res <> MMSYSERR_NOERROR then
  begin
    waveOutUnprepareHeader(fOut, h, SizeOf(TWaveHdr));
    FreeMem(h);
    raise ETtsAudio.CreateUtf8('waveOutWrite %', [res]);
  end;
  fQueue[(fHead + fCount) mod length(fQueue)] := h;
  inc(fCount);
  inc(fWritten, aBytes);
end;

function TTtsPcmStream.Drain(const aAbort: TOnTtsAbort): boolean;
begin
  result := true;
  Reclaim;
  while fCount > 0 do
  begin
    if Assigned(aAbort) and aAbort() then
    begin
      Stop;
      result := false;
      exit;
    end;
    WaitForSingleObject(fDone, 100);
    Reclaim;
  end;
end;

procedure TTtsPcmStream.Stop;
begin
  if fOut = 0 then
    exit;
  waveOutReset(fOut); // returns all the buffers at once
  Reclaim;
end;

function TTtsPcmStream.PendingMs: cardinal;
var
  i: PtrInt;
  bytes: Int64;
  h: PWaveHdr;
begin
  Reclaim;
  bytes := 0;
  for i := 0 to fCount - 1 do
  begin
    h := fQueue[(fHead + i) mod length(fQueue)];
    inc(bytes, h^.dwBufferLength);
  end;
  result := bytes div fBytesPerMs;
end;

{$endif OSWINDOWS}

end.
