import type * as ort from 'onnxruntime-web/jspi';

export async function readGpuFp16Bits(
  tensor: ort.Tensor,
  dims: readonly number[],
  name: string,
): Promise<Uint16Array> {
  if (
    tensor.type !== 'float16'
    || tensor.location !== 'gpu-buffer'
    || tensor.dims.length !== dims.length
    || tensor.dims.some((value, index) => value !== dims[index])
  ) throw new Error(`${name} must be a GPU-resident float16 tensor with shape [${dims.join(',')}]`);

  const data = await tensor.getData();
  if (!(data instanceof Uint16Array) && !(data instanceof Float16Array))
    throw new Error(`${name} downloader did not return FP16 storage`);
  const bits = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2).slice();
  const expectedValues = dims.reduce((total, value) => total * value, 1);
  if (bits.length !== expectedValues)
    throw new Error(`${name} downloader returned ${bits.length} values, expected ${expectedValues}`);
  return bits;
}
