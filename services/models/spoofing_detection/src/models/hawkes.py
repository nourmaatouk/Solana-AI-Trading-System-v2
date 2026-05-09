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

    def _neg_log_likelihood(self, theta: np.ndarray, events: List[np.ndarray], N: int) -> float:
        """Computes NLL for sum of 2 exponentials."""
        mu = np.clip(theta[:N], 1e-5, None)
        alpha1 = np.clip(theta[N : N + N*N], 0.0, None).reshape((N, N))
        alpha2 = np.clip(theta[N + N*N : N + 2*N*N], 0.0, None).reshape((N, N))

        # Focus strictly on cross-asset causality
        np.fill_diagonal(alpha1, 0.0)
        np.fill_diagonal(alpha2, 0.0)

        nll = 0.0
        
        # 1. Integral compensator part
        T_max = 0.0
        for i in range(N):
            if len(events[i]) > 0:
                T_max = max(T_max, events[i][-1])
        
        for i in range(N):
            # mu_i * T
            nll += mu[i] * T_max
            
            # sum_j (alpha1_ij/beta1 + alpha2_ij/beta2) * sum_k (1 - exp(-beta*(T-tk)))
            for j in range(N):
                if i != j and len(events[j]) > 0:
                    tj = events[j]
                    term1 = np.sum(1.0 - np.exp(-self.beta1 * (T_max - tj)))
                    term2 = np.sum(1.0 - np.exp(-self.beta2 * (T_max - tj)))
                    nll += (alpha1[i, j] / self.beta1) * term1
                    nll += (alpha2[i, j] / self.beta2) * term2
                        
        # 2. Log intensities at event times
        for i in range(N):
            ti = events[i]
            for t in ti:
                lam = mu[i]
                for j in range(N):
                    if i != j:
                        tj = events[j]
                        tj_past = tj[tj < t]
                        if len(tj_past) > 0:
                            # Sum over past events for both kernels
                            dt = t - tj_past
                            lam += alpha1[i, j] * np.sum(np.exp(-self.beta1 * dt))
                            lam += alpha2[i, j] * np.sum(np.exp(-self.beta2 * dt))
                
                nll -= np.log(max(lam, 1e-10))
                
        return nll

    def fit(self, event_times_dict: Dict[int, List[float]]) -> None:
        """Estimate parameters via MLE."""
        N = self.n_assets
        events = [np.sort(np.array(event_times_dict.get(i, []), dtype=np.float64)) for i in range(N)]

        # theta = [mu (N), alpha1 (N*N), alpha2 (N*N)]
        theta0 = np.concatenate([
            np.ones(N),          # mu
            np.full(N*N, 0.05),  # alpha1
            np.full(N*N, 0.05)   # alpha2
        ])
        
        bounds = [(1e-4, None)] * N + [(0.0, None)] * (2 * N * N)

        try:
            res = minimize(
                self._neg_log_likelihood, 
                theta0, 
                args=(events, N),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-3}
            )
            
            alpha1_opt = res.x[N : N + N*N].reshape((N, N))
            alpha2_opt = res.x[N + N*N : N + 2*N*N].reshape((N, N))
            
            # Combined Branching Ratio
            A = (alpha1_opt / self.beta1) + (alpha2_opt / self.beta2)

            # Row-max normalization (A / row_max) guarantees at least one 1.0 per row,
            # which saturates the GNN attention for every asset regardless of true coupling
            # strength.  Clip to [0, 1] instead — preserves relative magnitudes and
            # keeps super-critical rows (branching ratio > 1) at the ceiling.
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
