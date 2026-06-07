import numpy as np
import pickle
from hmmlearn.hmm import GMMHMM

class LatentRouter:
    def __init__(self, n_components=3, n_mix=2, covariance_type="diag", random_state=42):
        self.model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=100
        )
        self.state_map = {}
        self.is_fitted = False

    def fit(self, latent_X: np.ndarray):
        """Fits the GMM-HMM on the [N, 4] latent features"""
        self.model.fit(latent_X)
        self.is_fitted = True

    def _apply_heuristic_mapping(self, latent_X: np.ndarray, raw_returns: np.ndarray):
        """
        The Label Switching Defense.
        Mathematically anchors states to "TREND", "COMPRESSION", "MEAN_REV"
        based on the physical properties of the underlying price data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before mapping.")

        # Predict raw integer states (0, 1, 2)
        states = self.model.predict(latent_X)
        
        stats = {}
        for state_idx in range(self.model.n_components):
            mask = (states == state_idx)
            if np.sum(mask) == 0:
                # Fallback if state is empty (extremely rare with 3 states)
                stats[state_idx] = {"var": 1e9, "drift": 0}
                continue
            
            state_returns = raw_returns[mask]
            stats[state_idx] = {
                "var": np.var(state_returns),
                "drift": np.abs(np.mean(state_returns))
            }
        
        # 1. COMPRESSION: Lowest variance
        sorted_by_var = sorted(stats.items(), key=lambda x: x[1]["var"])
        comp_state = sorted_by_var[0][0]
        
        # Remove compression from consideration
        remaining_states = [s[0] for s in sorted_by_var[1:]]
        
        # 2. TREND: Highest absolute drift among the remaining two
        if stats[remaining_states[0]]["drift"] > stats[remaining_states[1]]["drift"]:
            trend_state = remaining_states[0]
            mean_rev_state = remaining_states[1]
        else:
            trend_state = remaining_states[1]
            mean_rev_state = remaining_states[0]
            
        self.state_map = {
            comp_state: "COMPRESSION",
            trend_state: "TREND",
            mean_rev_state: "MEAN_REV"
        }
        print(f"Heuristic Anchoring Complete: {self.state_map}")
        
    def predict_proba(self, latent_X: np.ndarray) -> tuple:
        """Returns the dictionary of state probabilities with locked labels"""
        if not self.state_map:
            raise ValueError("Heuristic mapping must be run before predicting.")
            
        probas = self.model.predict_proba(latent_X)
        
        mapped_probas = []
        for p in probas:
            mapped_probas.append({
                self.state_map[0]: p[0],
                self.state_map[1]: p[1],
                self.state_map[2]: p[2]
            })
        return mapped_probas

    def predict_causal_proba(self, latent_X: np.ndarray) -> list[dict]:
        """
        Calculates strictly causal filtered probabilities (alpha) without lookahead bias.
        """
        import scipy.special
        if not self.state_map:
            raise ValueError("Heuristic mapping must be run before predicting.")
            
        # 1. Compute log probabilities of each observation under each GMM emission state
        framelogprob = self.model._compute_log_likelihood(latent_X)
        
        # 2. Run ONLY the forward pass (alpha lattice)
        from hmmlearn import _hmmc
        logprob, fwdlattice = _hmmc.forward_log(self.model.startprob_, self.model.transmat_, framelogprob)
        
        # 3. Normalize the log probabilities at each time step t to sum to 1
        alpha = np.exp(fwdlattice - scipy.special.logsumexp(fwdlattice, axis=1, keepdims=True))
        
        mapped_probas = []
        for p in alpha:
            mapped_probas.append({
                self.state_map[0]: p[0],
                self.state_map[1]: p[1],
                self.state_map[2]: p[2]
            })
        return mapped_probas

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)
