import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_ribbon,
    theme_minimal, labs, theme, element_text, facet_wrap,
    coord_cartesian,
)

np.random.seed(42)
n = 200
x = np.sort(np.random.uniform(0, 5, n))

# -- Scenario 1: Low, constant variance (homoscedastic, low noise)
y1 = 0.3 * np.sin(2 * x) + np.random.normal(0, 0.3, n)
std1 = np.full_like(x, 0.6)

# -- Scenario 2: Heteroscedastic variance (grows in the middle)
noise2 = 0.3 + 2.5 * np.exp(-0.5 * ((x - 2.5) / 0.8) ** 2)
y2 = 0.3 * np.sin(2 * x) + np.random.normal(0, noise2, n)
std2 = noise2 * 2

# -- Scenario 3: High, constant variance (homoscedastic, high noise)
y3 = 0.3 * np.sin(2 * x) + np.random.normal(0, 2.0, n)
std3 = np.full_like(x, 4.0)

# -- Best predictor (same for all three)
y_pred = 0.3 * np.sin(2 * x)

# -- Build dataframe
rows = []
for xi, yi, pi, si, label in [
    (x, y1, y_pred, std1, "Scenario 1"),
    (x, y2, y_pred, std2, "Scenario 2"),
    (x, y3, y_pred, std3, "Scenario 3"),
]:
    for j in range(len(xi)):
        rows.append({
            "X": xi[j],
            "Y": yi[j],
            "pred": pi[j],
            "lower": pi[j] - si[j],
            "upper": pi[j] + si[j],
            "Scenario": label,
        })

df = pd.DataFrame(rows)

# -- Plot
p = (
    ggplot(df, aes(x="X"))
    + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="#90CAF9", alpha=0.5)
    + geom_point(aes(y="Y"), color="#E57373", alpha=0.6, size=1.2)
    + geom_line(aes(y="pred"), color="#FF8F00", size=1)
    + facet_wrap("~Scenario", ncol=3)
    + coord_cartesian(ylim=(-10, 10))
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
        figure_size=(12, 4),
    )
    + labs(
        x="X",
        y="Y",
        title="Same best predictor, 3 distinct scenarios",
    )
)

p.save("uncertainty_examples.pdf", width=11, height=6, dpi=200)

