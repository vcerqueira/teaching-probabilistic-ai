import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, geom_point,
    theme_minimal, labs, scale_color_identity, scale_alpha_identity
)

# -- Load California Housing dataset
data = fetch_california_housing(as_frame=True)
df_full = data.frame

feature = "MedInc"
target = "MedHouseVal"

# Subsample and sort for clean plotting
np.random.seed(42)
idx = np.random.choice(len(df_full), size=800, replace=False)
df_sub = df_full.iloc[idx].sort_values(feature).reset_index(drop=True)

X = df_sub[[feature]].values
y = df_sub[target].values

# -- Fit Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
rf.fit(X, y)

# -- Get individual tree predictions
tree_preds = np.array([tree.predict(X) for tree in rf.estimators_])

# Ensemble mean and std
mean_pred = tree_preds.mean(axis=0)
std_pred = tree_preds.std(axis=0)

# -- Build dataframe for individual trees (subsample 15 trees for visibility)
n_trees_to_show = 15
tree_indices = np.random.choice(len(rf.estimators_), n_trees_to_show, replace=False)

tree_rows = []
for i, t_idx in enumerate(tree_indices):
    for j in range(len(X)):
        tree_rows.append({
            feature: df_sub[feature].values[j],
            "prediction": tree_preds[t_idx, j],
            "tree": f"Tree {t_idx}",
        })
df_trees = pd.DataFrame(tree_rows)

# -- Build dataframe for ensemble summary
df_ensemble = pd.DataFrame({
    feature: df_sub[feature].values,
    target: y,
    "mean": mean_pred,
    "lower_1sd": mean_pred - std_pred,
    "upper_1sd": mean_pred + std_pred,
    "lower_2sd": mean_pred - 2 * std_pred,
    "upper_2sd": mean_pred + 2 * std_pred,
})

# -- Plot
p = (
    ggplot()
    # +/- 2 std band (light)
    + geom_ribbon(
        aes(x=feature, ymin="lower_2sd", ymax="upper_2sd"),
        data=df_ensemble, fill="#F44336", alpha=0.10,
    )
    # +/- 1 std band (darker)
    + geom_ribbon(
        aes(x=feature, ymin="lower_1sd", ymax="upper_1sd"),
        data=df_ensemble, fill="#F44336", alpha=0.20,
    )
    # Scatter of actual values
    + geom_point(
        aes(x=feature, y=target),
        data=df_ensemble, color="gray", alpha=0.3, size=0.8,
    )
    # Individual tree predictions
    + geom_line(
        aes(x=feature, y="prediction", group="tree"),
        data=df_trees, color="#EF9A9A", alpha=0.4, size=0.7,
    )
    # Ensemble mean
    + geom_line(
        aes(x=feature, y="mean"),
        data=df_ensemble, color="#B71C1C", size=1.5,
    )
    + theme_minimal()
    + labs(
        x="Median Income (x $10k)",
        y="Median House Value (x $100k)",
        title="Random Forest — Individual Trees vs. Ensemble",
        subtitle="15 individual trees (thin red), ensemble mean (bold), ±1σ and ±2σ bands",
    )
)

p.save("uncertainty_rf_plot.pdf", width=12, height=5, dpi=200)
