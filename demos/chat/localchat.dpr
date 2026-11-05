// Local console chat: native inference on an AMD Vulkan GPU or on the CPU.
// The model stays loaded and the conversation is kept until /reset or /exit.
program localchat;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef OSWINDOWS}
  windows,
  {$endif OSWINDOWS}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.core.unicode,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.infer.model.qwen3.vulkan;

type
  TChatPrinter = class
  public
    Content: RawUtf8;
    CompletionTokens: integer;
    procedure OnDelta(const aDelta: TLlmStreamDelta);
  end;

procedure TChatPrinter.OnDelta(const aDelta: TLlmStreamDelta);
begin
  if aDelta.HasUsage then
    CompletionTokens := aDelta.Usage.CompletionTokens;
  Content := Content + aDelta.ContentDelta;
  if aDelta.ContentDelta <> '' then
    ConsoleWrite(aDelta.ContentDelta, ccWhite, {nolinefeed=}true);
end;

function DurationText(aMicroseconds: Int64): RawUtf8;
begin
  if aMicroseconds < 1000000 then
    result := FormatUtf8('% ms', [(aMicroseconds + 500) div 1000])
  else if aMicroseconds < 60000000 then
    result := StringToUtf8(FormatFloat('0.00', aMicroseconds / 1000000)) + ' s'
  else
    result := FormatUtf8('% min % s', [aMicroseconds div 60000000,
      (aMicroseconds div 1000000) mod 60]);
end;

procedure Main;
var
  model: TFileName;
  prompt: RawUtf8;
  line: string;
  cpu, vulkan: boolean;
  maxTokens, maxContext, n: integer;
  backend: TInferBackend;
  client: TNativeLlmClient;
  llm: ILlmClient;
  printer: TChatPrinter;
  messages: TLlmMessageDynArray;
  loadStarted, loadFinished: Int64;

  procedure Ask(const aText: RawUtf8);
  var
    started, finished, elapsed: Int64;
  begin
    n := length(messages);
    SetLength(messages, n + 1);
    messages[n] := LlmMessage(lrUser, aText);
    printer.Content := '';
    printer.CompletionTokens := 0;
    ConsoleWrite('Assistant> ', ccLightGreen, {nolinefeed=}true);
    QueryPerformanceMicroSeconds(started);
    try
      llm.ChatStream(LlmChatRequest('', messages), printer.OnDelta);
    except
      SetLength(messages, n); // a failed request is not a conversation turn
      raise;
    end;
    QueryPerformanceMicroSeconds(finished);
    elapsed := finished - started;
    ConsoleWrite('');
    if (printer.CompletionTokens > 0) and (elapsed > 0) then
      ConsoleWrite('Generated in % | % tokens | % tokens/s overall', [
        DurationText(elapsed), printer.CompletionTokens,
        FormatFloat('0.0', printer.CompletionTokens * 1000000.0 / elapsed)])
    else
      ConsoleWrite('Generated in %', [DurationText(elapsed)]);
    SetLength(messages, n + 2);
    messages[n + 1] := LlmMessage(lrAssistant, printer.Content);
  end;

begin
  with Executable.Command do
  begin
    ExeDescription := 'localchat: local model on a Vulkan GPU or CPU';
    model := ParamS('model', 'the model #file.gguf or Hugging Face folder',
      ExpandFileName(ExtractFilePath(ParamStr(0)) +
        '..' + PathDelim + 'models' + PathDelim +
        'Qwen3-14B-Q4_K_M.gguf'));
    cpu := Option('cpu', 'compute on the CPU');
    vulkan := Option('vulkan', 'compute on the Vulkan GPU (default, AMD supported)');
    prompt := Param('prompt', 'answer this #text once, then exit');
    maxTokens := Param('tokens', 512, 'maximum generated #tokens per answer');
    maxContext := Param('context', 4096, 'maximum #tokens in the conversation');
    if ConsoleHelpFailed then
      exit;
  end;
  if cpu and vulkan then
    raise Exception.Create('Choose /cpu or /vulkan, not both.');
  if (maxTokens < 1) or (maxContext < 2) or (maxTokens >= maxContext) then
    raise Exception.Create('Require 0 < /tokens < /context.');
  if not FileExists(model) and not DirectoryExists(model) then
    raise Exception.Create('Model not found: ' + model +
      '. Use /model to select a GGUF file or a Hugging Face folder.');

  backend := nil;
  llm := nil;
  printer := TChatPrinter.Create;
  try
    QueryPerformanceMicroSeconds(loadStarted);
    if not cpu then
    begin
      backend := TInferVulkanBackend.Create;
      ConsoleWrite('Vulkan GPU: %',
        [TInferVulkanBackend(backend).Vulkan.DeviceName]);
    end
    else
      ConsoleWrite('Backend: CPU');
    ConsoleWrite('Loading model: %', [model]);
    if SameText(ExtractFileExt(model), '.gguf') then
      client := TNativeLlmClient.CreateFromGguf(model, '', backend)
    else
      client := TNativeLlmClient.Create(model, '', backend);
    llm := client; // client owns the model and device, backend outlives both
    if not cpu then
      client.Device := TQwen3Vulkan.Create(client.Model,
        TInferVulkanBackend(backend));
    client.EnableThinking := false;
    // Some Qwen3 GGUFs are non-thinking Instruct-2507 checkpoints. Only add
    // the synthetic no-thinking prefix if their original template supports
    // that switch; otherwise keep the normal assistant prefix.
    if (client.Model.Gguf <> nil) and
       client.Model.Gguf.Has('tokenizer.chat_template') and
       (PosEx('enable_thinking',
         client.Model.Gguf.Str('tokenizer.chat_template')) = 0) then
      client.EnableThinking := client.Family.Template.DefaultEnableThinking;
    client.MaxTokens := maxTokens;
    client.MaxContext := maxContext;
    client.Cache.Slots := 1;
    QueryPerformanceMicroSeconds(loadFinished);
    ConsoleWrite('Model loaded in %', [DurationText(loadFinished - loadStarted)]);

    if prompt <> '' then
    begin
      Ask(prompt);
      exit;
    end;
    ConsoleWrite('Ready. /reset clears the conversation; /exit quits.');
    while true do
    begin
      ConsoleWrite('You> ', ccLightBlue, {nolinefeed=}true);
      if Eof(Input) then
        break;
      ReadLn(line);
      prompt := TrimU(StringToUtf8(line));
      if prompt = '/exit' then
        break;
      if prompt = '/reset' then
      begin
        messages := nil;
        client.Cache.Slots := 0;
        client.Cache.Slots := 1;
        ConsoleWrite('Conversation cleared.');
      end
      else if prompt <> '' then
        try
          Ask(prompt);
        except
          on e: Exception do
            ConsoleWrite(#10'ERROR: %. Use /reset to start a new conversation.',
              [e.Message]);
        end;
    end;
  finally
    llm := nil;
    printer.Free;
    backend.Free;
  end;
end;

procedure RunConsole;
{$ifdef OSWINDOWS}
var
  inputCodePage, outputCodePage: cardinal;
{$endif OSWINDOWS}
begin
  {$ifdef OSWINDOWS}
  inputCodePage := GetConsoleCP;
  outputCodePage := GetConsoleOutputCP;
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
  SetTextCodePage(Input, CP_UTF8);
  try
  {$endif OSWINDOWS}
    Main;
  {$ifdef OSWINDOWS}
  finally
    if inputCodePage <> 0 then
      SetConsoleCP(inputCodePage);
    if outputCodePage <> 0 then
      SetConsoleOutputCP(outputCodePage);
  end;
  {$endif OSWINDOWS}
end;

begin
  try
    RunConsole;
  except
    on e: Exception do
    begin
      ConsoleShowFatalException(e, {waitforenterkey=}false);
      ExitCode := 2;
    end;
  end;
end.
