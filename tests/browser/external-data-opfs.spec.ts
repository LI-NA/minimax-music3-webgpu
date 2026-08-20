import { expect, test } from '@playwright/test';

test('JSPI reads external OPFS File ranges without materializing the File', async ({ page }) => {
  await page.goto('/');
  const output = await page.evaluate(async () => {
    const moduleUrl = '/src/runtime/browser-smokes.ts';
    return (await import(/* @vite-ignore */ moduleUrl)).runExternalDataOpfsSmoke();
  });
  expect(output).toBe(5);
});
