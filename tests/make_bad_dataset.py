import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "id": range(n),
    "constant_col": [5] * n,
    "feature_a": np.random.normal(50, 10, n),
    "feature_b": None,
    "target": np.random.choice([0, 1], n, p=[0.95, 0.05]),
})

df["feature_b"] = df["feature_a"] * 2 + 1

df.loc[0:30, "feature_a"] = np.nan

df = pd.concat([df, df.iloc[0:10]], ignore_index=True)

df.loc[5, "feature_a"] = 5000

df.to_csv("data/synthetic_bad.csv", index=False)
print("Synthetic bad dataset created.")