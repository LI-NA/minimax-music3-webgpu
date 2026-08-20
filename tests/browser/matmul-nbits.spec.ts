import { expect, test } from '@playwright/test';

test('WebGPU symmetric q4 MatMulNBits matches the frozen known answer', async ({ page }) => {
  await page.goto('/');
  const values = await page.evaluate(async () => {
    const moduleUrl = '/src/runtime/browser-smokes.ts';
    return (await import(/* @vite-ignore */ moduleUrl)).runMatMulNBitsSmoke();
  });
  expect(values).toHaveLength(2);
  expect(Math.abs(values[0] - 128)).toBeLessThan(0.02);
  expect(Math.abs(values[1] - 256)).toBeLessThan(0.02);
});
