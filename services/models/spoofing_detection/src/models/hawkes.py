"""
hawkes.py — Hawkes Process Adjacency Matrix Estimator
======================================================

Estimates multi-asset directional causality (lead-lag relationships) using 
a multivariate Hawkes Process with a sum of 2 exponential kernels.

Kernel: phi_ij(t) = alpha1_ij * exp(-beta1 * t) + alpha2_ij * exp(-beta2 * t)
  - beta1 = 10.0 (fast decay — captures microsecond reactions)
  - beta2 = 0.5  (slow decay — captures multi-second clustering)

The branching ratio A_ij = alpha1_ij/beta1 + alpha2_ij/beta2 forms the adjacency matrix.
"""

import logging
import threading
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class HawkesEstimator:
    """Thread-safe multivariate Hawkes Process estimator with double-exponential kernel.
    
    Parameters
    ----------
    n_assets : int
        Number of assets in the graph.
    beta1 : float, default 10.0
        Fast decay rate.
    beta2 : float, default 0.5
        Slow decay rate.
    """

    def __init__(self, n_assets: int, beta1: float = 10.0, beta2: float = 0.5):
        self.n_assets = n_assets
        self.beta1 = beta1
        self.beta2 = beta2
        
        self._lock = threading.Lock()
        self._adjacency = np.eye(n_assets, dtype=np.float32)

    # ── Vectorised helpers ────────────────────────────────────────────────────

    @staticmethod
    def _precompute_S(tj: np.ndarray, beta: float) -> np.ndarray:
        """Precompute S[k] = Σ_{l≤k} exp(-β·(tj[k]-tj[l])) via the recurrence
        S[k] = exp(-β·dt_k)·S[k-1] + 1.

        Computed once per (asset, kernel) pair before optimisation starts, so the
        O(T) sequential loop runs only twice per asset (not once per gradient call).
        Numerically stable: dt_k ≥ 0 so exp terms decay, S stays bounded.
        """
        if len(tj) == 0:
            return np.empty(0)
        S = np.ones(len(tj))
        for k in range(1, len(tj)):
            S[k] = np.exp(-beta * (tj[k] - tj[k - 1])) * S[k - 1] + 1.0
        return S

    @staticmethod
    def _R_at_ti(ti: np.ndarray, tj: np.ndarray, S_j: np.ndarray, beta: float) -> np.ndarray:
        """Vectorised cross-asset excitation R_ij at each event time in ti.

        R_ij(t_i^n) = Σ_{t_j^k < t_i^n} exp(-β·(t_i^n - t_j^k))
                    = exp(-β·(t_i^n - tj[last])) · S_j[last]
        where 'last' is the index of the most recent j-event before t_i^n.
        """
        if len(tj) == 0 or len(ti) == 0:
            return np.zeros(len(ti))
        idx = np.searchsorted(tj, ti, side="left") - 1   # last t_j^k < t_i^n
        result = np.zeros(len(ti))
        mask = idx >= 0
        if mask.any():
            k = idx[mask]
            result[mask] = np.exp(-beta * (ti[mask] - tj[k])) * S_j[k]
        return result

    # ── NLL + analytical gradient (vectorised, one pass per optimizer step) ──

    def _nll_and_grad(
        self,
        theta: np.ndarray,
        events: List[np.ndarray],
        N: int,
        S1: List[np.ndarray],
        S2: List[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        """Returns (NLL, gradient) using vectorised cross-asset excitation.

        Key speedup: S1[j] / S2[j] are precomputed once before optimisation (they
        don't depend on parameters).  Each gradient call then replaces the O(T)
        Python event-loop with N² vectorised np.searchsorted + array-multiply
        calls, cutting per-call cost from O(T·Python-overhead) to O(N²·numpy).

        Gradient:
          ∂NLL/∂μ_i       = T - Σ_{t_i^n} 1/λ_i(t_i^n)
          ∂NLL/∂α1_{ij}   = (1/β1)·Σ_{t_j}(1-exp(-β1(T-t_j))) - Σ_{t_i^n} R1_ij(t_i^n)/λ_i
          (analogously for α2, β2)
        """
        mu    = np.clip(theta[:N], 1e-5, None)
        alpha1 = np.clip(theta[N     : N + N*N],    0.0, None).reshape((N, N))
        alpha2 = np.clip(theta[N+N*N : N + 2*N*N],  0.0, None).reshape((N, N))
        np.fill_diagonal(alpha1, 0.0)
        np.fill_diagonal(alpha2, 0.0)

        non_empty = [ev for ev in events if len(ev) > 0]
        if not non_empty:
            return 0.0, np.zeros_like(theta)

        T_max = max(ev[-1] for ev in non_empty)

        nll  = float(np.sum(mu) * T_max)
        g_mu = np.full(N, T_max, dtype=np.float64)
        g_a1 = np.zeros((N, N), dtype=np.float64)
        g_a2 = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            ti = events[i]
            if len(ti) == 0:
                continue

            # Accumulate λ_i at each t_i^n and the per-pair R matrices
            lam_i  = np.full(len(ti), mu[i], dtype=np.float64)
            R1_store: List[np.ndarray] = [np.empty(0)] * N
            R2_store: List[np.ndarray] = [np.empty(0)] * N

            for j in range(N):
                if i == j or len(events[j]) == 0:
                    continue
                tj = events[j]

                # Integral compensator
                dt_j = T_max - tj
                c1 = float(np.sum(1.0 - np.exp(-self.beta1 * dt_j)))
                c2 = float(np.sum(1.0 - np.exp(-self.beta2 * dt_j)))
                nll       += (alpha1[i, j] / self.beta1) * c1
                nll       += (alpha2[i, j] / self.beta2) * c2
                g_a1[i, j] += c1 / self.beta1
                g_a2[i, j] += c2 / self.beta2

                # Vectorised cross-excitation
                r1 = self._R_at_ti(ti, tj, S1[j], self.beta1)
                r2 = self._R_at_ti(ti, tj, S2[j], self.beta2)
                R1_store[j] = r1
                R2_store[j] = r2
                lam_i += alpha1[i, j] * r1 + alpha2[i, j] * r2

            inv_lam = 1.0 / np.maximum(lam_i, 1e-10)
            nll    -= float(np.sum(np.log(np.maximum(lam_i, 1e-10))))
            g_mu[i] -= float(np.sum(inv_lam))

            for j in range(N):
                if i != j and len(R1_store[j]) > 0:
                    g_a1[i, j] -= float(np.sum(R1_store[j] * inv_lam))
                    g_a2[i, j] -= float(np.sum(R2_store[j] * inv_lam))

        grad = np.concatenate([g_mu, g_a1.ravel(), g_a2.ravel()])
        return nll, grad

    def fit(self, event_times_dict: Dict[int, List[float]]) -> None:
        """Estimate parameters via MLE using vectorised NLL + analytical gradient.

        S1[j] / S2[j] are precomputed once (they don't depend on parameters) so
        the optimiser only pays the O(T) Python precompute cost once, not once per
        gradient call.  Each gradient call is then dominated by N² vectorised numpy
        operations on arrays of length T_i, not by T Python iterations.
        """
        N = self.n_assets
        events = [np.sort(np.array(event_times_dict.get(i, []), dtype=np.float64)) for i in range(N)]

        # Precompute recursive sums — O(T) Python loop, run once
        S1 = [self._precompute_S(ev, self.beta1) for ev in events]
        S2 = [self._precompute_S(ev, self.beta2) for ev in events]

        theta0 = np.concatenate([
            np.ones(N),          # mu
            np.full(N*N, 0.05),  # alpha1
            np.full(N*N, 0.05),  # alpha2
        ])
        bounds = [(1e-4, None)] * N + [(0.0, None)] * (2 * N * N)

        try:
            res = minimize(
                self._nll_and_grad,
                theta0,
                args=(events, N, S1, S2),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-4},
            )
            
            alpha1_opt = res.x[N : N + N*N].reshape((N, N))
            alpha2_opt = res.x[N + N*N : N + 2*N*N].reshape((N, N))

            # Zero diagonal: optimizer may have non-zero values for diagonal
            # parameters even though they carry no gradient (zeroed inside NLL).
            np.fill_diagonal(alpha1_opt, 0.0)
            np.fill_diagonal(alpha2_opt, 0.0)

            # Combined Branching Ratio — clipped to [0, 0.9] (sub-critical)
            A = (alpha1_opt / self.beta1) + (alpha2_opt / self.beta2)
            A = np.clip(A, 0.0, 0.9)

            with self._lock:
                self._adjacency = A.astype(np.float32)
                
            logger.debug("Hawkes adjacency updated (Double-Exp Kernel).")
            
        except Exception as e:
            logger.warning(f"Hawkes MLE failed: {e}. Retaining previous adjacency.")

    def get_adjacency(self) -> np.ndarray:
        with self._lock:
            return self._adjacency.copy()

def compare_kernels():
    """Comparison test between Single and Double Exponential kernels."""
    n_assets = 2
    # Leader-Follower pattern with some 'after-effect'
    events_0 = np.array([0.1, 0.2, 0.5, 0.8])
    events_1 = np.concatenate([
        events_0 + 0.01,  # Fast reaction
        events_0 + 0.3    # Slow follow-up
    ])
    events_1.sort()
    
    event_dict = {0: events_0.tolist(), 1: events_1.tolist()}

    print("\n--- Kernel Comparison ---")
    
    # 1. Single Exp
    class SingleExp(HawkesEstimator):
        def _neg_log_likelihood(self, theta, events, N):
            mu, alpha = theta[:N], theta[N:].reshape((N, N))
            np.fill_diagonal(alpha, 0.0)
            nll = 0.0
            T_max = max([e[-1] for e in events if len(e)>0])
            for i in range(N):
                nll += mu[i] * T_max
                for j in range(N):
                    if i != j and len(events[j]) > 0:
                        nll += (alpha[i, j] / self.beta1) * np.sum(1.0 - np.exp(-self.beta1 * (T_max - events[j])))
            for i in range(N):
                for t in events[i]:
                    lam = mu[i] + sum(alpha[i, j] * np.sum(np.exp(-self.beta1 * (t - events[j][events[j] < t]))) for j in range(N) if i!=j)
                    nll -= np.log(max(lam, 1e-10))
            return nll
        def fit(self, event_times_dict):
            N = self.n_assets
            events = [np.sort(np.array(event_times_dict.get(i, []), dtype=np.float64)) for i in range(N)]
            theta0 = np.concatenate([np.ones(N), np.full(N*N, 0.1)])
            res = minimize(self._neg_log_likelihood, theta0, args=(events, N), method='L-BFGS-B', bounds=[(1e-4, None)]*(N+N*N))
            A = res.x[N:].reshape((N,N)) / self.beta1
            self._adjacency = (A / np.maximum(A.max(axis=1, keepdims=True), 1e-2)).astype(np.float32)

    se = SingleExp(n_assets=2, beta1=10.0)
    se.fit(event_dict)
    A_single = se.get_adjacency()
    
    # 2. Double Exp
    de = HawkesEstimator(n_assets=2, beta1=10.0, beta2=0.5)
    de.fit(event_dict)
    A_double = de.get_adjacency()
    
    print(f"Single-exp A_ij:\n{np.round(A_single, 3)}")
    print(f"Double-exp A_ij:\n{np.round(A_double, 3)}")
    
    if A_double[1,0] > A_single[1,0]:
        print("✓ Double-exp captured higher cross-asset influence (A[1,0]).")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    compare_kernels()
