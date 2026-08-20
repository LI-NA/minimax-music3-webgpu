import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(60 * 60_000);
const frameCount = Number(process.env.MINIMAX_FRAMES ?? '2');

test('generates two exact-checkpoint RVQ frames with WebGPU-resident autoregressive state', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_CHROME_PROFILE ?? 'artifacts/autoregressive-browser-profile'),
    { channel: 'chrome', headless: false },
  );
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto(`http://127.0.0.1:5173/?frames=${frameCount}`);
    await page.getByRole('button', { name: 'Generate RVQ frames' }).click();
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="frame-generation-result"]') ||
        document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 60 * 60_000 },
    );
    await expect(page.locator('output')).not.toHaveText(/^Error:/);
    const result = page.getByTestId('frame-generation-result');
    await expect(result).toContainText(`frames: ${frameCount}`);
    await expect(result).toContainText(`semantic decisions: ${frameCount + 1}`);
    await expect(result).toContainText(`RVQ calls: ${(frameCount + 1) * 7}`);
    await expect(result).toContainText(`feedback decodes: ${frameCount}`);
    await expect(result).toContainText('cache lengths: 40');
    await expect(result).toContainText(`, ${40 + frameCount}`);
    await expect(result).toContainText('finite hidden groups: yes');
    await expect(result).toContainText('codes in range: yes');
  } finally {
    await context.close();
  }
});
