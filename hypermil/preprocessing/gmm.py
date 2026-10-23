"""Regularized full-covariance EM and its numerical precision fallback."""
from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.cluster import KMeans


class RegularizedGMM:
    def __init__(self, K=10, covariance="full", concentration=0.1, min_component_mass=1e-3,
                 max_iter=100, tol=1e-3, seed=42, device="cpu"):
        if covariance not in {"full", "diag"}: raise ValueError(covariance)
        self.K, self.covariance, self.concentration = K, covariance, concentration
        self.min_component_mass, self.max_iter, self.tol = min_component_mass, max_iter, tol
        self.seed, self.device = seed, torch.device(device)
        self.iterations_run = 0
        self.final_log_likelihood = None
        self.previous_log_likelihood = None
        self.delta_log_likelihood = None
        self.converged = False
        self.component_effective_counts = []
        self.nan_count = 0
        self.inf_count = 0
        self.runtime_seconds = 0.0

    def _initialize(self, x):
        n, d = x.shape; k_eff = min(self.K, n)
        labels = KMeans(k_eff, n_init=1, random_state=self.seed).fit_predict(x.cpu().numpy())
        means = []
        for k in range(self.K):
            members = x[labels == (k % k_eff)]
            means.append(members.mean(0) if members.shape[0] else x[k % n].clone())
        self.means = torch.stack(means)
        centered = x - x.mean(0)
        global_full = centered.T @ centered / max(n, 1) + 1e-4 * torch.eye(d, device=x.device)
        self.covariances = (global_full.diag().repeat(self.K, 1) if self.covariance == "diag"
                            else global_full.repeat(self.K, 1, 1))
        self.weights = torch.full((self.K,), 1.0 / self.K, device=x.device)

    def _log_prob(self, x):
        n, d = x.shape; out = x.new_empty((n, self.K))
        for k in range(self.K):
            diff = x - self.means[k]
            if self.covariance == "diag":
                var = self.covariances[k].clamp_min(1e-6)
                out[:, k] = -0.5 * (d * np.log(2*np.pi) + var.log().sum() + (diff.square()/var).sum(1))
            else:
                cov = 0.5 * (self.covariances[k] + self.covariances[k].T)
                eye = torch.eye(d, device=x.device, dtype=x.dtype)
                chol = None
                jitter = 1e-6
                for _ in range(6):
                    candidate, info = torch.linalg.cholesky_ex(cov + jitter * eye)
                    if int(info.item()) == 0:
                        chol = candidate
                        break
                    jitter *= 10.0
                if chol is None:
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                    repaired = (eigenvectors * eigenvalues.clamp_min(1e-4)) @ eigenvectors.T
                    chol = torch.linalg.cholesky(0.5 * (repaired + repaired.T) + 1e-6 * eye)
                solve = torch.linalg.solve_triangular(chol, diff.T, upper=False)
                out[:, k] = -0.5 * (d*np.log(2*np.pi) + 2*chol.diag().log().sum() + solve.square().sum(0))
        return out + self.weights.clamp_min(1e-12).log()

    def fit_predict_proba(self, features):
        started = time.perf_counter()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim != 2 or x.shape[0] == 0: raise ValueError("features must be non-empty [N,D]")
        self._initialize(x); previous = None
        self.converged = False
        for iteration in range(self.max_iter):
            logp = self._log_prob(x); gamma = torch.softmax(logp, 1); ll = torch.logsumexp(logp, 1).sum()
            nk = gamma.sum(0); min_mass = max(float(self.min_component_mass), 1e-8)
            for k in range(self.K):
                if nk[k] < min_mass:
                    distances = torch.cdist(x, self.means).min(1).values
                    idx = int(torch.argmax(distances).item()); self.means[k] = x[idx]
                    gamma[:, k] = 0; gamma[idx, k] = min_mass; nk[k] = min_mass
            gamma = gamma / gamma.sum(1, keepdim=True).clamp_min(1e-12); nk = gamma.sum(0).clamp_min(1e-10)
            self.weights = (nk + self.concentration) / (x.shape[0] + self.K*self.concentration)
            self.means = gamma.T @ x / nk[:, None]
            for k in range(self.K):
                diff = x - self.means[k]
                if self.covariance == "diag":
                    self.covariances[k] = (gamma[:, k, None]*diff.square()).sum(0)/nk[k] + 1e-6
                else:
                    self.covariances[k] = (diff.T*gamma[:, k])@diff/nk[k] + 1e-6*torch.eye(x.shape[1],device=x.device)
            self.iterations_run = iteration + 1
            self.previous_log_likelihood = None if previous is None else float(previous)
            self.final_log_likelihood = float(ll)
            self.delta_log_likelihood = None if previous is None else float(ll - previous)
            if previous is not None and abs(self.delta_log_likelihood) <= self.tol * max(1.0, abs(float(previous))):
                self.converged = True
                break
            previous = ll.detach()
        gamma = torch.softmax(self._log_prob(x), 1)
        self.component_effective_counts = [float(value) for value in gamma.sum(0).detach().cpu()]
        self.nan_count = int(torch.isnan(gamma).sum().item())
        self.inf_count = int(torch.isinf(gamma).sum().item())
        self.runtime_seconds = time.perf_counter() - started
        if self.nan_count or self.inf_count: raise FloatingPointError("non-finite responsibilities")
        return gamma

    def convergence_record(self, sample_id=None):
        return {
            "sample_id": sample_id,
            "K": self.K,
            "covariance_type": self.covariance,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "criterion": "abs(L_t-L_t-1) <= tol * max(1, abs(L_t-1))",
            "iterations_run": self.iterations_run,
            "final_log_likelihood": self.final_log_likelihood,
            "previous_log_likelihood": self.previous_log_likelihood,
            "delta_log_likelihood": self.delta_log_likelihood,
            "converged": self.converged,
            "component_effective_counts": self.component_effective_counts,
            "component_utilization": sum(value > self.min_component_mass for value in self.component_effective_counts),
            "NaN_count": self.nan_count,
            "Inf_count": self.inf_count,
            "runtime": self.runtime_seconds,
        }


class Float64RegularizedGMM(RegularizedGMM):
    """Numerically equivalent full-covariance fallback for ill-conditioned slides."""
    def fit_predict_proba(self,features):
        started=time.perf_counter();x=torch.as_tensor(np.asarray(features),dtype=torch.float64,device=self.device)
        if x.ndim!=2 or x.shape[0]==0:raise ValueError("features must be non-empty [N,D]")
        self._initialize(x);previous=None;self.converged=False
        for iteration in range(self.max_iter):
            logp=self._log_prob(x);gamma=torch.softmax(logp,1);ll=torch.logsumexp(logp,1).sum()
            if not torch.isfinite(gamma).all():raise FloatingPointError("float64 fallback produced non-finite responsibilities")
            nk=gamma.sum(0);min_mass=max(float(self.min_component_mass),1e-8)
            for k in range(self.K):
                if nk[k]<min_mass:
                    distances=torch.cdist(x,self.means).min(1).values;idx=int(torch.argmax(distances));self.means[k]=x[idx]
                    gamma[:,k]=0;gamma[idx,k]=min_mass;nk[k]=min_mass
            gamma=gamma/gamma.sum(1,keepdim=True).clamp_min(1e-15);nk=gamma.sum(0).clamp_min(1e-15)
            self.weights=(nk+self.concentration)/(x.shape[0]+self.K*self.concentration);self.means=gamma.T@x/nk[:,None]
            for k in range(self.K):
                diff=x-self.means[k]
                self.covariances[k]=(diff.T*gamma[:,k])@diff/nk[k]+1e-6*torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
            self.iterations_run=iteration+1;self.previous_log_likelihood=None if previous is None else float(previous)
            self.final_log_likelihood=float(ll);self.delta_log_likelihood=None if previous is None else float(ll-previous)
            if previous is not None and abs(self.delta_log_likelihood)<=self.tol*max(1.0,abs(float(previous))):self.converged=True;break
            previous=ll.detach()
        gamma=torch.softmax(self._log_prob(x),1)
        self.component_effective_counts=[float(v) for v in gamma.sum(0).detach().cpu()]
        self.nan_count=int(torch.isnan(gamma).sum());self.inf_count=int(torch.isinf(gamma).sum());self.runtime_seconds=time.perf_counter()-started
        if self.nan_count or self.inf_count:raise FloatingPointError("non-finite responsibilities")
        return gamma.float()
