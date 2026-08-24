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
    'Upbeat K-pop dance-pop, 124 BPM, major key.',
    'Bright and energetic with a glossy, modern radio-ready mix.',
    'Punchy drums, bouncy synth bass, sparkling plucks and airy pads.',
  ].join('\n'),
  vocal: [
    'Bright female lead vocal, clean and agile with a light sweet tone.',
    'Rhythmic, playful phrasing in the verses, wide layered harmonies in the chorus.',
    'Doubled hook lines with short chant-style ad-libs and a touch of bright reverb.',
  ].join('\n'),
  arrangement: [
    'The intro opens with a catchy synth hook over four-on-the-floor drums.',
    'The verse grooves on a bouncy bass line with tight hi-hats and claps.',
    'The pre-chorus lifts with rising synths, building drums and a vocal run.',
    'The chorus bursts open with full drums, bright synth stabs and the chanted hook.',
  ].join('\n'),
  instrumentalMeta: [
    'Energetic EDM, progressive house, 128 BPM.',
    'Euphoric festival-ready sound with a loud, punchy club mix.',
    'Driving four-on-the-floor kick, sidechained supersaw leads, deep rolling bass.',
  ].join('\n'),
  instrumentalVocal: 'Instrumental. No vocals.',
  instrumentalArrangement: [
    'The intro sets a filtered synth loop over a steady four-on-the-floor kick.',
    'The build-up layers rising white noise, snare rolls and a pitch-climbing lead.',
    'The drop hits with massive supersaw chords, sidechain pumping and heavy bass.',
    'The outro strips back to the filtered loop and fades on a long reverb tail.',
  ].join('\n'),
  lyrics: [
    '[Verse]',
    '반짝이는 불빛 사이로 두근대는 맘',
    '오늘 밤엔 멈추지 않아 리듬에 맡겨봐',
    '',
    '[Chorus]',
    'Turn it up, turn it up, 더 크게 소리쳐',
    '빛나는 우리 이 순간, never let it go',
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
