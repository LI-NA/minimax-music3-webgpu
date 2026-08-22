import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import promptContract from '../../fixtures/prompt-contract.json';
import fixedCase from '../../../tools/reference/fixed_case.json';
import {
  createPromptTokenizer,
  preparePrompt,
  type PromptTokenizer,
} from '../../../src/runtime/pipeline/prompt-preparation';

const fixtureRows = [promptContract.conditional, promptContract.unconditional] as const;

const fixedPrompt = fixedCase.prompt;
const fixedLyrics = fixedCase.lyrics;
const tokenizerPath = 'artifacts/release/music-variable/global/tokenizer/tokenizer.json';
const configPath = 'artifacts/release/music-variable/global/tokenizer/tokenizer_config.json';
const hasLocalReleaseTokenizer = existsSync(tokenizerPath) && existsSync(configPath);

function tokenizerFor(ids: readonly number[]) {
  let text = '';
  const tokenizer: PromptTokenizer = {
    encode(value) {
      text = value;
      return { ids };
    },
  };
  return { tokenizer, text: () => text };
}

describe('prompt preparation', () => {
  it('assembles the exact fixed checkpoint prompt and CFG rows', () => {
    const fake = tokenizerFor(fixtureRows[0]);

    expect(
      preparePrompt({
        prompt: fixedPrompt,
        lyrics: fixedLyrics,
        requestedFrames: 250,
        tokenizer: fake.tokenizer,
      }),
    ).toEqual({
      assembledPrompt: fixedCase.assembledPrompt,
      promptTokens: 40,
      tokenRows: fixtureRows,
    });
    expect(fake.text()).toBe(fixedCase.assembledPrompt);
  });

  it('reproduces the pinned caption and lyric preprocessing rules', () => {
    const fake = tokenizerFor([151644, 10, 151645, 151669]);

    const prepared = preparePrompt({
      prompt: ' ## **<|genre Jazz|>**\r\n- *Vocal*  \r\n---\r\n• four    spaces',
      lyrics: ' [Verse] discarded words\n[ChOrUs] [Bridge] still discarded\nA ] B [C] ^ D',
      requestedFrames: 1,
      tokenizer: fake.tokenizer,
    });

    expect(prepared.assembledPrompt).toBe(
      '<|im_start|><|caption_start|>genre is Jazz\nVocal\nfourspaces<|caption_end|>' +
        '<|lyrics_start|>[start]\n[verse]\n[chorus]\n[bridge]\nA ]\nB\n[c]\n^ D<|lyrics_end|>' +
        '<|im_end|><|audio_start|>',
    );
  });

  it('rejects blank inputs, oversized prompts, and context overflows', () => {
    const fake = tokenizerFor([151644, 151645, 151669]);
    expect(() =>
      preparePrompt({
        prompt: ' ',
        lyrics: 'lyric',
        requestedFrames: 1,
        tokenizer: fake.tokenizer,
      }),
    ).toThrow('prompt');
    expect(() =>
      preparePrompt({
        prompt: 'prompt',
        lyrics: '\t',
        requestedFrames: 1,
        tokenizer: fake.tokenizer,
      }),
    ).toThrow('lyrics');
    expect(() =>
      preparePrompt({
        prompt: 'prompt',
        lyrics: 'lyric',
        requestedFrames: 1,
        tokenizer: tokenizerFor(Array(5_001).fill(1)).tokenizer,
      }),
    ).toThrow('5000');
    expect(() =>
      preparePrompt({
        prompt: 'prompt',
        lyrics: 'lyric',
        requestedFrames: 7_500,
        tokenizer: tokenizerFor(Array(2_741).fill(1)).tokenizer,
      }),
    ).toThrow('10240');
  });

  it.skipIf(!hasLocalReleaseTokenizer)('requires the local release tokenizer to encode the fixed case', async () => {
    const [tokenizerJson, tokenizerConfigJson] = await Promise.all([
      readFile(tokenizerPath, 'utf8'),
      readFile(configPath, 'utf8'),
    ]);
    const tokenizer = createPromptTokenizer(tokenizerJson, tokenizerConfigJson);

    expect(
      preparePrompt({
        prompt: fixedPrompt,
        lyrics: fixedLyrics,
        requestedFrames: 250,
        tokenizer,
      }),
    ).toEqual({
      assembledPrompt: fixedCase.assembledPrompt,
      promptTokens: 40,
      tokenRows: fixtureRows,
    });
  });
});
