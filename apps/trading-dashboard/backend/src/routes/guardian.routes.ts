/**
 * Guardian policy routes. Exposes the active policy thresholds for the
 * dashboard. Read-only — policy is configured via .env, not via HTTP.
 */
import { Router } from 'express';
import { GuardianValidator } from '@trading/execution';

export const guardianRouter = Router();

guardianRouter.get('/policy', (_req, res) => {
  const policy = new GuardianValidator().describePolicy();
  res.json({
    policy,
    notes: [
      'Policy values are sourced from environment variables.',
      'To change a threshold, update .env and restart the backend.',
      'The persistent emergency_stop flag overrides env at runtime.',
    ],
  });
});
