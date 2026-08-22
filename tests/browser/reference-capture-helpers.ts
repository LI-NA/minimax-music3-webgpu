import path from 'node:path';
import { canonicalContainedChild } from './qualification-paths';

export type ReferenceCaptureLayout = {
  captureId: string;
  captureRoot: string;
  captureDirectory: string;
  caseRoot: string;
  caseDirectory: string;
  profileRoot: string;
  profile: string;
};

export type CanonicalWavAnalysis = {
  structure: {
    riff: 'RIFF';
    wave: 'WAVE';
    format: 1;
    channels: 2;
    sampleRate: 44100;
    byteRate: 176400;
    blockAlign: 4;
    bitsPerSample: 16;
    riffSize: number;
    dataBytes: number;
    samplesPerChannel: number;
    wavBytes: number;
  };
  health: {
    stereoDiffers: boolean;
    longestConstantFrameRun: number;
    finalSecondDelta: number;
  };
};

export function resolveCaptureLayout(
  checkoutRoot: string,
  captureId: string,
  requestedProfile?: string,
  linkedWorktree = false,
): ReferenceCaptureLayout {
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(captureId))
    throw new Error('capture id must contain only lowercase letters, digits, and hyphens');
  const artifactRoot = path.resolve(checkoutRoot, 'artifacts');
  const captureRoot = path.join(artifactRoot, 'reference', 'captures');
  const caseRoot = path.join(artifactRoot, 'reference', 'cases');
  const profileRoot = path.join(artifactRoot, linkedWorktree ? 'worktree-profiles' : 'browser-profiles');
  const captureDirectory = path.join(captureRoot, captureId);
  const caseDirectory = path.join(caseRoot, captureId);
  if (linkedWorktree && !requestedProfile)
    throw new Error('linked worktrees require an explicit reference Chrome profile');
  const profile = requestedProfile
    ? path.resolve(checkoutRoot, requestedProfile)
    : path.join(profileRoot, 'variable-duration', 'task11');
  if (!canonicalContainedChild(captureRoot, captureDirectory) || !canonicalContainedChild(caseRoot, caseDirectory))
    throw new Error('capture and case paths must remain inside their artifact roots');
  if (!canonicalContainedChild(profileRoot, profile)) {
    const rootName = linkedWorktree ? 'worktree-profiles' : 'browser-profiles';
    throw new Error(`reference Chrome profile must remain inside artifacts/${rootName}`);
  }
  return {
    captureId,
    captureRoot,
    captureDirectory,
    caseRoot,
    caseDirectory,
    profileRoot,
    profile,
  };
}

export function assertFreshCaptureLayout(layout: ReferenceCaptureLayout, exists: (candidate: string) => boolean) {
  if (exists(layout.captureDirectory)) throw new Error(`reference capture already exists: ${layout.captureDirectory}`);
  if (exists(layout.caseDirectory)) throw new Error(`reference case already exists: ${layout.caseDirectory}`);
}

export function assertMatchingWavSha256(first: string, second: string) {
  if (first !== second) throw new Error('fixed WebGPU WAV is not byte reproducible');
}

const ascii = (bytes: Uint8Array, offset: number, length: number) =>
  String.fromCharCode(...bytes.subarray(offset, offset + length));

export function analyzeCanonicalPcm16Wav(bytes: Uint8Array): CanonicalWavAnalysis {
  if (bytes.byteLength < 48) throw new Error('WAV must be canonical PCM16 stereo');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dataBytes = view.getUint32(40, true);
  const canonical =
    ascii(bytes, 0, 4) === 'RIFF' &&
    view.getUint32(4, true) === bytes.byteLength - 8 &&
    ascii(bytes, 8, 8) === 'WAVEfmt ' &&
    view.getUint32(16, true) === 16 &&
    view.getUint16(20, true) === 1 &&
    view.getUint16(22, true) === 2 &&
    view.getUint32(24, true) === 44_100 &&
    view.getUint32(28, true) === 176_400 &&
    view.getUint16(32, true) === 4 &&
    view.getUint16(34, true) === 16 &&
    ascii(bytes, 36, 4) === 'data' &&
    dataBytes === bytes.byteLength - 44 &&
    dataBytes % 4 === 0;
  if (!canonical) throw new Error('WAV must be canonical PCM16 stereo at 44,100 Hz');

  const samplesPerChannel = dataBytes / 4;
  if (samplesPerChannel < 1) throw new Error('WAV must contain at least one sample frame');
  let stereoDiffers = false;
  let longestConstantFrameRun = 1;
  let currentConstantFrameRun = 1;
  let finalSecondDelta = 0;
  const finalSecondStart = Math.max(1, samplesPerChannel - 44_100);
  let previousLeft = view.getInt16(44, true);
  let previousRight = view.getInt16(46, true);
  for (let frame = 1; frame < samplesPerChannel; frame++) {
    const offset = 44 + frame * 4;
    const left = view.getInt16(offset, true);
    const right = view.getInt16(offset + 2, true);
    stereoDiffers ||= left !== right;
    if (left === previousLeft && right === previousRight) currentConstantFrameRun++;
    else currentConstantFrameRun = 1;
    longestConstantFrameRun = Math.max(longestConstantFrameRun, currentConstantFrameRun);
    if (frame >= finalSecondStart) finalSecondDelta += Math.abs(left - previousLeft) + Math.abs(right - previousRight);
    previousLeft = left;
    previousRight = right;
  }
  stereoDiffers ||= previousLeft !== previousRight;

  return {
    structure: {
      riff: 'RIFF',
      wave: 'WAVE',
      format: 1,
      channels: 2,
      sampleRate: 44_100,
      byteRate: 176_400,
      blockAlign: 4,
      bitsPerSample: 16,
      riffSize: bytes.byteLength - 8,
      dataBytes,
      samplesPerChannel,
      wavBytes: bytes.byteLength,
    },
    health: { stereoDiffers, longestConstantFrameRun, finalSecondDelta },
  };
}
