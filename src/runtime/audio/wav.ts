const HEADER_BYTES = 44;
const CHANNELS = 2;
const BYTES_PER_SAMPLE = 2;

export function encodeStereoPcm16Wav(
  channels: readonly [Float32Array, Float32Array],
  sampleRate = 44_100,
): ArrayBuffer {
  const [left, right] = channels;
  if (left.length !== right.length) throw new Error('channel lengths must match');
  if (!Number.isSafeInteger(sampleRate) || sampleRate <= 0)
    throw new Error('sample rate must be a positive integer');
  const dataBytes = left.length * CHANNELS * BYTES_PER_SAMPLE;
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
  for (let index = 0; index < left.length; index++) {
    writeSample(view, HEADER_BYTES + index * 4, left[index]);
    writeSample(view, HEADER_BYTES + index * 4 + 2, right[index]);
  }
  return buffer;
}

function writeSample(view: DataView, offset: number, sample: number): void {
  if (!Number.isFinite(sample)) throw new Error('samples must be finite');
  const value = Math.max(-1, Math.min(1, sample));
  view.setInt16(offset, Math.round(value * (value < 0 ? 32_768 : 32_767)), true);
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index++) view.setUint8(offset + index, value.charCodeAt(index));
}
