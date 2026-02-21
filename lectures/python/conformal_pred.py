"""
Conformal prediction: barplot of calibration scores (sorted) and 90% cutoff threshold.
Threshold q̂ is the ⌈(1-α)(1 + n_cal)⌉-th smallest score (here α=0.1 for 90% coverage).
"""
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_col,
    geom_hline,
    geom_ribbon,
    geom_line,
    geom_point,
    facet_wrap,
    theme_minimal,
    labs,
    scale_fill_manual,
    theme,
    element_text,
)

np.random.seed(42)
n_cal = 50
alpha = 0.1  # 90% coverage

# Synthetic calibration scores (e.g. nonconformity scores 1 - P(Y=y|x))
scores = np.sort(np.random.beta(2, 5, size=n_cal))

# Conformal threshold: k = ceil((1-alpha)(n_cal + 1)), q̂ = k-th smallest score
k = int(np.ceil((1 - alpha) * (n_cal + 1)))
q_hat = scores[k - 1]

df = pd.DataFrame({
    "index": np.arange(1, n_cal + 1),
    "score": scores,
    "above_threshold": scores > q_hat,
})

p = (
        ggplot(df, aes(x="index", y="score", fill="above_threshold"))
        + geom_col(width=0.7)
        + geom_hline(yintercept=q_hat, color="#C62828", linetype="dashed", size=1.2)
        + scale_fill_manual(values={True: "#EF9A9A", False: "#64B5F6"})
        + theme_minimal()
        + theme(
    legend_position="none",
    figure_size=(8, 4),
)
        + labs(
    x="Calibration instance (scores sorted)",
    y="Score s",
    title="Conformal threshold (90% coverage)",
    subtitle=f"Threshold q-hat = {q_hat:.3f} (k = ceil((1-alpha)(1+n_cal)) = {k}-th smallest; alpha = {alpha})",
)
)

p.save("conformal_scores_threshold.pdf", width=8, height=4, dpi=200)

# ---------------------------------------------------------------------------
# Conformal prediction intervals: constant vs adaptive width (scatter, side by side)
# ---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(123)
n = 400
x = np.sort(np.random.uniform(0, 10, n))
# Heteroscedastic: noise grows with x so adaptive width is visibly different
noise_scale = 0.3 + 0.25 * x
y = 1.5 * np.sin(0.6 * x) + 0.3 * x + np.random.normal(0, noise_scale, n)
X = x.reshape(-1, 1)

# Train / calibration split
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.25, random_state=42)
n_cal_reg = len(y_cal)
alpha_reg = 0.1

# Fit mean predictor
model = LinearRegression().fit(X_train, y_train)
y_pred_cal = model.predict(X_cal)
residuals_abs = np.abs(y_cal - y_pred_cal)

# Constant-width: score = |y - ŷ(x)|, threshold q̂
k_reg = int(np.ceil((1 - alpha_reg) * (n_cal_reg + 1)))
q_constant = np.sort(residuals_abs)[k_reg - 1]

# Adaptive-width: score = |y - ŷ(x)| / σ̂(x). Estimate σ̂(x) from calibration.
# Fit |residual| ~ x to get scale as function of x
scale_model = LinearRegression().fit(X_cal, residuals_abs)
sigma_cal = np.maximum(scale_model.predict(X_cal), 1e-6)
scores_adaptive = residuals_abs / sigma_cal
q_adaptive = np.sort(scores_adaptive)[k_reg - 1]

# Plot grid and intervals
x_plot = np.linspace(0, 10, 200)
X_plot = x_plot.reshape(-1, 1)
y_plot = model.predict(X_plot)
sigma_plot = np.maximum(scale_model.predict(X_plot), 1e-6)

lower_const = y_plot - q_constant
upper_const = y_plot + q_constant
lower_adapt = y_plot - q_adaptive * sigma_plot
upper_adapt = y_plot + q_adaptive * sigma_plot

# Long format for facet: Constant | Adaptive
df_const = pd.DataFrame({
    "x": x_plot,
    "y": y_plot,
    "lower": lower_const,
    "upper": upper_const,
    "method": "Constant width",
})
df_adapt = pd.DataFrame({
    "x": x_plot,
    "y": y_plot,
    "lower": lower_adapt,
    "upper": upper_adapt,
    "method": "Adaptive width",
})
df_bands = pd.concat([df_const, df_adapt], ignore_index=True)

# Scatter points (use calibration for visibility)
df_pts = pd.DataFrame({"x": X_cal.ravel(), "y": y_cal})
df_pts["method"] = "Constant width"
df_pts2 = df_pts.copy()
df_pts2["method"] = "Adaptive width"
df_pts_both = pd.concat([df_pts, df_pts2], ignore_index=True)

p2 = (
        ggplot()
        + geom_ribbon(data=df_bands, mapping=aes(x="x", ymin="lower", ymax="upper"), fill="#64B5F6", alpha=0.4)
        + geom_line(data=df_bands, mapping=aes(x="x", y="y"), color="#1565C0", size=1)
        + geom_point(data=df_pts_both, mapping=aes(x="x", y="y"), color="#E57373", alpha=0.6, size=1.5)
        + facet_wrap("~method", ncol=2)
        + theme_minimal()
        + theme(figure_size=(10, 4.5), strip_text=element_text(size=11, weight="bold"))
        + labs(x="x", y="y", title="Conformal prediction intervals (90% coverage)")
)
p2.save("conformal_intervals_constant_vs_adaptive.pdf", width=10, height=4.5, dpi=200)

# ---------------------------------------------------------------------------
# Trade-off: coverage (x) vs interval width (y). Trivially wide at 100% coverage.
# ---------------------------------------------------------------------------
sorted_res = np.sort(residuals_abs)
coverages = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
widths = []
for cov in coverages:
    alpha_t = 1.0 - cov
    k_t = int(np.ceil((1 - alpha_t) * (n_cal_reg + 1)))
    k_t = min(k_t, n_cal_reg)  # cap at n_cal for 100% (use max residual)
    q_t = sorted_res[k_t - 1]
    widths.append(2 * q_t)

df_trade = pd.DataFrame({"coverage": coverages, "width": widths})

p3 = (
        ggplot(df_trade, aes(x="coverage", y="width"))
        + geom_line(color="#1565C0", size=1.2)
        + geom_point(color="#1565C0", size=3)
        + theme_minimal()
        + theme(figure_size=(7, 4))
        + labs(
    x="Coverage (1 − α)",
    y="Interval width (2q̂)",
    title="Coverage vs interval width",
    subtitle="Higher coverage -> wider intervals; 100% coverage -> trivially wide interval",
)
)
p3.save("conformal_coverage_width_tradeoff.pdf", width=7, height=4, dpi=200)
