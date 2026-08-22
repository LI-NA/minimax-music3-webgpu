import { createHash } from 'node:crypto';

const sourceBytes = 16_020_739;
const sourceSha256 = '4e7ef2c42e9004207d111b1416f1ba69c89fba447f4d9628f6055578dd8e9297';
const patchedSha256 = '0569a267c57da3947fefc95934a7eee1426188cba11997be556515d482347534';
const brokenCast = new TextEncoder().encode('dy_element_t(');
const fixedCast = new TextEncoder().encode('f32         (');

const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

export function patchOrtWebGpuConvTransposeCoordinates(source: Uint8Array): Uint8Array {
  if (source.byteLength !== sourceBytes || hash(source) !== sourceSha256)
    throw new Error('onnxruntime-web JSPI WASM differs from the pinned runtime');

  const offsets: number[] = [];
  for (let offset = 0; offset <= source.length - brokenCast.length; offset++) {
    if (brokenCast.every((value, index) => source[offset + index] === value)) offsets.push(offset);
  }
  if (offsets.length !== 8) throw new Error(`expected 8 ConvTranspose coordinate casts, found ${offsets.length}`);

  const patched = Uint8Array.from(source);
  for (const offset of offsets) patched.set(fixedCast, offset);
  if (hash(patched) !== patchedSha256) throw new Error('patched onnxruntime-web JSPI WASM hash is unexpected');
  return patched;
}
