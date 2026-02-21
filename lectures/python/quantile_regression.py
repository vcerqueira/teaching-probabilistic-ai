import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_ribbon,
    theme_minimal, labs, scale_color_manual, scale_fill_manual
)

# -- Generate synthetic data with heteroscedastic noise
np.random.seed(42)
n = 300
x = np.sort(np.random.uniform(0, 10, n))
noise_scale = 0.5 + 0.3 * x  # noise grows with x
y = 2 * np.sin(x) + 0.5 * x + np.random.normal(0, noise_scale, n)

X = x.reshape(-1, 1)

# -- Fit quantile regression models
quantiles = {"q05": 0.05, "median": 0.50, "q95": 0.95}
predictions = {}

for name, alpha in quantiles.items():
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)
    predictions[name] = model.predict(X)

# -- Build dataframe for plotting
df = pd.DataFrame({
    "x": x,
    "y": y,
    "median": predictions["median"],
    "q05": predictions["q05"],
    "q95": predictions["q95"],
})

# -- Plot
p = (
    ggplot(df, aes(x="x"))
    + geom_ribbon(aes(ymin="q05", ymax="q95"), fill="#2196F3", alpha=0.25)
    + geom_point(aes(y="y"), color="gray", alpha=0.5, size=1.5)
    + geom_line(aes(y="median"), color="#1565C0", size=1.2)
    + geom_line(aes(y="q05"), color="#1565C0", size=0.5, linetype="dashed")
    + geom_line(aes(y="q95"), color="#1565C0", size=0.5, linetype="dashed")
    + theme_minimal()
    + labs(
        x="x",
        y="y",
        title="Quantile Regression",
        subtitle="Median prediction with 90% prediction interval (5th–95th quantile)",
    )
)

p.save("quantile_regression_plot.png", width=8, height=5, dpi=200)