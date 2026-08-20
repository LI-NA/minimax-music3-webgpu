import { expect, test } from '@playwright/test';

test.setTimeout(30 * 60_000);

test('prefills and performs ten GPU-resident cached decodes', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Run Global LLM smoke' }).click();

  const result = page.getByTestId('global-smoke-result');
  const progress = page.locator('output');
  await page.waitForFunction(
    () => document.querySelector('[data-testid="global-smoke-result"]') || document.querySelector('output')?.textContent?.startsWith('Error:'),
    undefined,
    { timeout: 30 * 60_000 },
  );
  await expect(progress).not.toHaveText(/^Error:/);
  await expect(result).toContainText('steps: 10', { timeout: 30 * 60_000 });
  await expect(result).toContainText('finite logits: yes');
  await expect(result).toContainText('KV location: gpu-buffer');
  await page.getByRole('button', { name: 'Run Global LLM smoke' }).click();
  await expect(result).toContainText('"artifactFetches": 0', { timeout: 30 * 60_000 });
});
