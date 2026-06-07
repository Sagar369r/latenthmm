import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from v4_engine.gmm_hmm import LatentRouter

def test_heuristic_mapping():
    # 1. Create a fake [Batch, 4] Latent tensor
    batch_size = 500
    # Create 3 distinct spatial clusters so the GMM-HMM can actually separate them
    latent_X = np.zeros((batch_size, 4))
    latent_X[0:150] = np.random.normal(loc=0.0, scale=1.0, size=(150, 4))    # Cluster 0
    latent_X[150:300] = np.random.normal(loc=10.0, scale=1.0, size=(150, 4)) # Cluster 1
    latent_X[300:500] = np.random.normal(loc=-10.0, scale=1.0, size=(200, 4))# Cluster 2
    
    # 2. Create synthetic raw returns that map cleanly to our heuristics
    raw_returns = np.zeros(batch_size)
    
    # Split the fake data into 3 segments to simulate HMM clusters natively
    # Cluster 0: High Variance, Low Drift -> MEAN_REV
    # Cluster 1: Low Variance -> COMPRESSION
    # Cluster 2: High Variance, High Drift -> TREND
    
    raw_returns[0:150] = np.random.normal(loc=0.000, scale=0.02, size=150)   # MEAN_REV
    raw_returns[150:300] = np.random.normal(loc=0.000, scale=0.002, size=150) # COMPRESSION
    raw_returns[300:500] = np.random.normal(loc=0.015, scale=0.015, size=200) # TREND
    
    router = LatentRouter(n_components=3, n_mix=2)
    
    # Fit the router
    router.fit(latent_X)
    
    # Run the heuristic mapper
    router._apply_heuristic_mapping(latent_X, raw_returns)
    
    # Check the state map contains all required labels
    assert "COMPRESSION" in router.state_map.values()
    assert "TREND" in router.state_map.values()
    assert "MEAN_REV" in router.state_map.values()
    
    # 3. Test probabilities output shape
    probas = router.predict_causal_proba(latent_X)
    assert len(probas) == batch_size
    assert "TREND" in probas[0]
    
    print("✓ GMM-HMM Router: Latent clustering and Heuristic Anchoring successful.")

if __name__ == "__main__":
    test_heuristic_mapping()
