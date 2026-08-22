const HEADER_BYTES = 44;
const CHANNELS = 2;
const BYTES_PER_SAMPLE = 2;

export function encodeStereoPcm16Wav(
  channels: readonly [Float32Array, Float32Array],
  sampleRate = 44_100,
): ArrayBuffer {
  const [left, right] = channels;
  if (left.length !== right.length) throw new Error('channel lengths must match');
  const buffer = createStereoPcm16Wav(left.length, sampleRate);
  writeStereoPcm16WavChannel(buffer, 0, 0, left);
  writeStereoPcm16WavChannel(buffer, 1, 0, right);
  return buffer;
}

export function createStereoPcm16Wav(samplesPerChannel: number, sampleRate = 44_100): ArrayBuffer {
  if (!Number.isSafeInteger(samplesPerChannel) || samplesPerChannel < 0)
    throw new Error('samples per channel must be a non-negative integer');
  if (!Number.isSafeInteger(sampleRate) || sampleRate <= 0) throw new Error('sample rate must be a positive integer');
  const dataBytes = samplesPerChannel * CHANNELS * BYTES_PER_SAMPLE;
  if (dataBytes > 0xffff_ffff - 36) throw new Error('WAV data exceeds RIFF capacity');
  const buffer = new ArrayBuffer(HEADER_BYTES + dataBytes);
  const view = new DataView(buffer);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * CHANNELS * BYTES_PER_SAMPLE, true);
  view.setUint16(32, CHANNELS * BYTES_PER_SAMPLE, true);
  view.setUint16(34, BYTES_PER_SAMPLE * 8, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);
  return buffer;
}

export function writeStereoPcm16WavChannel(
  buffer: ArrayBuffer,
  channel: 0 | 1,
  sampleOffset: number,
  samples: Float32Array,
): void {
  if (channel !== 0 && channel !== 1) throw new Error('channel must be 0 or 1');
  if (!Number.isSafeInteger(sampleOffset) || sampleOffset < 0)
    throw new Error('sample offset must be a non-negative integer');
  const capacity = (buffer.byteLength - HEADER_BYTES) / (CHANNELS * BYTES_PER_SAMPLE);
  if (!Number.isSafeInteger(capacity) || sampleOffset + samples.length > capacity)
    throw new Error('channel write exceeds WAV sample capacity');
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index++) {
    writeSample(
      view,
      HEADER_BYTES + (sampleOffset + index) * CHANNELS * BYTES_PER_SAMPLE + channel * BYTES_PER_SAMPLE,
      samples[index],
    );
  }
}

function writeSample(view: DataView, offset: number, sample: number): void {
  if (!Number.isFinite(sample)) throw new Error('samples must be finite');
  const value = Math.max(-1, Math.min(1, sample));
  view.setInt16(offset, Math.round(value * (value < 0 ? 32_768 : 32_767)), true);
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index++) view.setUint8(offset + index, value.charCodeAt(index));
}
