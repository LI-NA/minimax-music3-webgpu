const WAV_HEADER_BYTES = 44;
const BYTES_PER_FRAME = 4;

export function pseudoBars(seed: number, count: number): number[] {
  let state = seed >>> 0 || 1;
  const random = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return Array.from({ length: count }, (_, index) =>
    Math.round(12 + 78 * (0.4 * random() + 0.6 * Math.abs(Math.sin(index * 0.31 + (seed % 17))))),
  );
}

export async function wavBars(wav: Blob, count: number): Promise<number[]> {
  const buffer = await wav.arrayBuffer();
  const frames = Math.floor((buffer.byteLength - WAV_HEADER_BYTES) / BYTES_PER_FRAME);
  if (frames < count) return Array.from({ length: count }, () => 8);
  const view = new DataView(buffer);
  const framesPerBar = frames / count;
  const stride = Math.max(1, Math.floor(framesPerBar / 200));
  const peaks = Array.from({ length: count }, (_, bar) => {
    const start = Math.floor(bar * framesPerBar);
    const end = Math.floor((bar + 1) * framesPerBar);
    let peak = 0;
    for (let frame = start; frame < end; frame += stride) {
      const offset = WAV_HEADER_BYTES + frame * BYTES_PER_FRAME;
      peak = Math.max(peak, Math.abs(view.getInt16(offset, true)));
    }
    return peak;
  });
  const loudest = Math.max(1, ...peaks);
  return peaks.map((peak) => Math.round(8 + 84 * (peak / loudest)));
}
