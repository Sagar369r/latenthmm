import numpy as np
import pandas as pd
import time
X = np.random.randn(60000, 6)
df = pd.DataFrame(X)
t0 = time.time()
p1 = df.expanding(min_periods=2).quantile(0.01)
t1 = time.time()
print(f"Pandas expanding quantile: {t1-t0:.3f}s")
