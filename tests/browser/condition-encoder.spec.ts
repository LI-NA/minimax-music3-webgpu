import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(10 * 60_000);

test('runs the fixed condition encoder on WebGPU and reuses its OPFS release', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_CONDITION_CHROME_PROFILE ?? 'artifacts/browser-profile-condition'),
    { channel: 'chrome', headless: false },
  );
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto('http://127.0.0.1:5173/diagnostics/condition');
    await page.getByRole('button', { name: 'Run condition encoder smoke' }).click({ timeout: 5_000 });

    const result = page.getByTestId('condition-smoke-result');
    const progress = page.locator('output');
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="condition-smoke-result"]') ||
        document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 10 * 60_000 },
    );
    await expect(progress).not.toHaveText(/^Error:/);
    await expect(result).toContainText('shape: 1, 430, 2048');
    await expect(result).toContainText('output location: gpu-buffer');
    await expect(result).toContainText('finite: yes');

    await page.getByRole('button', { name: 'Run condition encoder smoke' }).click();
    await expect(result).toHaveCount(0);
    await expect(result).toContainText('"artifactFetches": 0', { timeout: 10 * 60_000 });
  } finally {
    await context.close();
  }
});
