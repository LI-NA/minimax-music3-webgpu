import { Tokenizer } from '@huggingface/tokenizers';

const IM_START = '<|im_start|>';
const IM_END = '<|im_end|>';
const CAPTION_START = '<|caption_start|>';
const CAPTION_END = '<|caption_end|>';
const LYRICS_START = '<|lyrics_start|>';
const LYRICS_END = '<|lyrics_end|>';
const AUDIO_START = '<|audio_start|>';
const AUDIO_CFG_TOKEN_ID = 151_654;
const MAX_PROMPT_TOKENS = 5_000;
const MAX_CONTEXT_TOKENS = 10_240;

const specialTag = /<\|([^|]*)\|>/g;
const leadingTags = /^[ \t]*((?:\[[^\]]+\][ \t]*)+)/;
const lineBreak = new RegExp(
  `\\r\\n|[\\n\\v\\f\\r${String.fromCharCode(0x1c)}-${String.fromCharCode(0x1e)}\\x85\\u2028\\u2029]`,
);

export type PromptTokenizer = {
  encode(text: string): { ids: readonly number[] };
};

export type PromptTokenizerAdapter = {
  create(tokenizerJson: object, tokenizerConfigJson: object): PromptTokenizer;
};

export type PromptPreparationInput = {
  prompt: unknown;
  lyrics: unknown;
  requestedFrames: unknown;
  tokenizer: PromptTokenizer;
};

export type PreparedPrompt = {
  assembledPrompt: string;
  promptTokens: number;
  tokenRows: [number[], number[]];
};

export const tokenizersJsAdapter: PromptTokenizerAdapter = {
  create: (tokenizerJson, tokenizerConfigJson) => new Tokenizer(tokenizerJson, tokenizerConfigJson),
};

function jsonObject(text: string, label: string): object {
  const value: unknown = JSON.parse(text);
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    throw new Error(`${label} must be a JSON object`);
  return value;
}

export function createPromptTokenizer(
  tokenizerJson: string,
  tokenizerConfigJson: string,
  adapter: PromptTokenizerAdapter = tokenizersJsAdapter,
): PromptTokenizer {
  return adapter.create(
    jsonObject(tokenizerJson, 'tokenizer.json'),
    jsonObject(tokenizerConfigJson, 'tokenizer_config.json'),
  );
}

export function cleanCaption(caption: string): string {
  const text = caption.replace(specialTag, (_match, contents: string) => {
    const inner = contents.trim();
    const parts = /^(\S+)(?:\s+(.*))?$/.exec(inner);
    return parts?.[2] === undefined ? inner : `${parts[1]} is ${parts[2]}`;
  });
  const lines = text.split(lineBreak).map((source) => {
    let line = source.replace(/^\s{0,3}#{1,6}\s+/, '');
    line = line.replace(/^\s*[*+-]\s+/, '');
    line = line.replace(/^\s*\*\s+/, '');
    while (line.includes('**')) {
      const updated = line.replace(/\*\*([^*]+)\*\*/, '$1');
      if (updated === line) break;
      line = updated;
    }
    return line.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '$1').replace(/\s+$/, '');
  });
  return lines
    .join('\n')
    .replace(/^\s*[-*_]{3,}\s*$/gm, '')
    .replaceAll('• ', '')
    .replaceAll('    ', '')
    .replace(/\n{2,}/g, '\n');
}

export function normalizeLyrics(lyrics: string): string {
  const text = lyrics
    .split('\n')
    .map((line) => {
      const match = leadingTags.exec(line);
      return match ? match[1].trim() : line;
    })
    .join('\n');
  return `[start]\n${text
    .replaceAll('] ', ']\n')
    .replaceAll(' [', '\n[')
    .replaceAll(' ^ ', '\n')
    .replace(/\[([^\]]+)\]/g, (_match, contents: string) => `[${contents.toLowerCase()}]`)}`;
}

function validateText(value: unknown, label: 'prompt' | 'lyrics'): asserts value is string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string`);
}

function validateFrames(requestedFrames: unknown): asserts requestedFrames is number {
  if (typeof requestedFrames !== 'number' || !Number.isSafeInteger(requestedFrames) || requestedFrames < 0)
    throw new Error('requested frames must be a non-negative safe integer');
}

function validateIds(ids: readonly number[], requestedFrames: number): void {
  if (ids.length < 3) throw new Error('assembled prompt must contain at least three tokens');
  if (ids.length > MAX_PROMPT_TOKENS)
    throw new Error(`The assembled prompt has ${ids.length} tokens; the maximum is ${MAX_PROMPT_TOKENS}`);
  if (ids.length + requestedFrames > MAX_CONTEXT_TOKENS)
    throw new Error(`Prompt tokens plus frames must not exceed ${MAX_CONTEXT_TOKENS}`);
}

export function preparePrompt({ prompt, lyrics, requestedFrames, tokenizer }: PromptPreparationInput): PreparedPrompt {
  validateText(prompt, 'prompt');
  validateText(lyrics, 'lyrics');
  validateFrames(requestedFrames);
  const assembledPrompt =
    `${IM_START}${CAPTION_START}${cleanCaption(prompt)}${CAPTION_END}` +
    `${LYRICS_START}${normalizeLyrics(lyrics)}${LYRICS_END}${IM_END}${AUDIO_START}`;
  const conditional = [...tokenizer.encode(assembledPrompt).ids];
  validateIds(conditional, requestedFrames);
  const unconditional = [...conditional];
  unconditional.fill(AUDIO_CFG_TOKEN_ID, 1, -2);
  return {
    assembledPrompt,
    promptTokens: conditional.length,
    tokenRows: [conditional, unconditional],
  };
}
