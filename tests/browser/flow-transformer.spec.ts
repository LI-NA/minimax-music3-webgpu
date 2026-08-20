import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(30 * 60_000);

test('runs one and 30 fixed flow steps on WebGPU and reuses its OPFS release', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_FLOW_CHROME_PROFILE ?? 'artifacts/browser-profile-flow'),
    { channel: 'chrome', headless: false },
  );
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto('http://127.0.0.1:5173/diagnostics/flow');
    await page.getByRole('button', { name: 'Run flow transformer smoke' }).click({ timeout: 5_000 });

    const result = page.getByTestId('flow-smoke-result');
    const progress = page.locator('output');
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="flow-smoke-result"]')
        || document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 30 * 60_000 },
    );
    await expect(progress).not.toHaveText(/^Error:/);
    await expect(result).toContainText('one-step shape: 1, 128, 430');
    await expect(result).toContainText('one-step location: gpu-buffer');
    await expect(result).toContainText('one-step finite: yes');
    await expect(result).toContainText('steps: 30');
    await expect(result).toContainText('final location: gpu-buffer');
    await expect(result).toContainText('final finite: yes');

    await page.getByRole('button', { name: 'Run flow transformer smoke' }).click();
    await expect(result).toHaveCount(0);
    await expect(result).toContainText('"artifactFetches": 0', { timeout: 30 * 60_000 });
  } finally {
    await context.close();
  }
});
