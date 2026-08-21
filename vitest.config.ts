import { defineConfig } from 'vitest/config';
import { appBuildDefines } from './vite.config.ts';

export default defineConfig({
  define: appBuildDefines,
  test: {
    include: ['tests/unit/**/*.test.ts'],
  },
});
