import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      // Mirror the tsconfig "paths" mapping so vitest resolves
      // `@trading/execution` the same way the runtime does.
      '@trading/execution': path.resolve(__dirname, '../../../services/execution/dist/index.js'),
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
    globals: false,
    environment: 'node',
  },
});
