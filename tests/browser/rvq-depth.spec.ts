import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(30 * 60_000);

test('runs all seven RVQ depths and feedback on WebGPU only', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_CHROME_PROFILE ?? 'artifacts/rvq-browser-profile'),
    { channel: 'chrome', headless: false },
  );
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto('http://127.0.0.1:5173/');
    await page.getByRole('button', { name: 'Run RVQ depth smoke' }).click();
    const result = page.getByTestId('rvq-smoke-result');
    const progress = page.locator('output');
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="rvq-smoke-result"]') ||
        document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 30 * 60_000 },
    );
    await expect(progress).not.toHaveText(/^Error:/);
    await expect(result).toContainText('lengths: 2, 3, 4, 5, 6, 7, 8');
    await expect(result).toContainText('finite logits: yes');
    await expect(result).toContainText('hidden location: gpu-buffer');
    await expect(result).toContainText('feedback location: gpu-buffer');
  } finally {
    await context.close();
  }
});
