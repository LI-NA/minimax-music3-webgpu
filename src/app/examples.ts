import type { MusicSamplingInput } from '../workers/protocol';

export const DEFAULT_SAMPLING: MusicSamplingInput = {
  globalGuidance: 1.5,
  semanticTopK: 50,
  residualTopK: 50,
  temperature: 1,
  flowGuidance: 1.7,
  flowSteps: 30,
};

export const EXAMPLES = {
  meta: [
    'Dream pop and shoegaze, 92 BPM, minor key.',
    'Dreamy and intimate at the beginning, gradually becoming wide and uplifting.',
    'Warm analog texture, soft high frequencies, deep but restrained bass.',
  ].join('\n'),
  vocal: [
    'Soft female lead vocal with a breathy lower register.',
    'Intimate delivery in the verses, opening into layered harmonies in the chorus.',
    'Light plate reverb and subtle delay on phrase endings.',
  ].join('\n'),
  arrangement: [
    'The intro begins with filtered guitar and a distant pad.',
    'The verse adds a restrained bass pulse and minimal drums.',
    'The chorus opens with full drums, wide guitars and layered synths.',
    'The bridge drops back to voice and piano before the final chorus.',
  ].join('\n'),
  instrumentalMeta: [
    'Ambient electronic, 60 BPM.',
    'Slowly evolving granular textures over a deep sub pulse.',
    'Spacious, cinematic, tactile.',
  ].join('\n'),
  instrumentalVocal: 'Instrumental. No vocals.',
  lyrics: [
    '[Verse]',
    'City lights dissolve into the rain',
    "Your voice is a signal I can't explain",
    '',
    '[Chorus]',
    "Hold the frequency, don't let go",
    'We are the static turning into glow',
  ].join('\n'),
};

export const LYRIC_TAGS = [
  'Intro',
  'Verse',
  'Pre-Chorus',
  'Chorus',
  'Post-Chorus',
  'Bridge',
  'Instrumental',
  'Solo',
  'Outro',
];
