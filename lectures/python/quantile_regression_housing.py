import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_ribbon,
    theme_minimal, labs
)

# -- Load California Housing dataset
data = fetch_california_housing(as_frame=True)
df_full = data.frame

# Use MedInc (median income) as the single feature for visualization
feature = "MedInc"
target = "MedHouseVal"

# Subsample for cleaner plot
np.random.seed(42)
idx = np.random.choice(len(df_full), size=1000, replace=False)
df_sub = df_full.iloc[idx].sort_values(feature).reset_index(drop=True)

X = df_sub[[feature]].values
y = df_sub[target].values

# -- Fit quantile regression models
quantiles = {"q05": 0.05, "q25": 0.25, "median": 0.50, "q75": 0.75, "q95": 0.95}
predictions = {}

for name, alpha in quantiles.items():
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X, y)
    predictions[name] = model.predict(X)

# -- Build dataframe for plotting
df_plot = pd.DataFrame({
    feature: df_sub[feature].values,
    target: y,
    "median": predictions["median"],
    "q05": predictions["q05"],
    "q25": predictions["q25"],
    "q75": predictions["q75"],
    "q95": predictions["q95"],
})

# -- Plot
p = (
    ggplot(df_plot, aes(x=feature))
    + geom_ribbon(aes(ymin="q05", ymax="q95"), fill="#2196F3", alpha=0.15)
    + geom_ribbon(aes(ymin="q25", ymax="q75"), fill="#2196F3", alpha=0.30)
    + geom_point(aes(y=target), color="gray", alpha=0.35, size=1)
    + geom_line(aes(y="median"), color="#1565C0", size=1.2)
    + geom_line(aes(y="q05"), color="#1565C0", size=0.4, linetype="dashed")
    + geom_line(aes(y="q95"), color="#1565C0", size=0.4, linetype="dashed")
    + theme_minimal()
    + labs(
        x="Median Income (x $10k)",
        y="Median House Value (x $100k)",
        title="Quantile Regression — California Housing",
        subtitle="Median with 50% (dark) and 90% (light) prediction bands",
    )
)

p.save("quantile_regression_housing.png", width=8, height=5, dpi=200)
print(p)