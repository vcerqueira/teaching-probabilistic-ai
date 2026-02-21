import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_col,
    facet_wrap,
    theme_minimal,
    labs,
    scale_fill_manual,
    theme,
    element_text, element_blank
)

# Long-format data: instance type, class, probability
data = [
    # Confident instance -> "predict"
    ("predict", "Class 1", 0.02),
    ("predict", "Class 2", 0.03),
    ("predict", "Class 3", 0.95),
    # Uncertain instance -> "reject"
    ("reject", "Class 1", 0.31),
    ("reject", "Class 2", 0.35),
    ("reject", "Class 3", 0.34),
]
df = pd.DataFrame(data, columns=["instance", "class", "probability"])

# Optional: fix order of classes and instance panels
df["class"] = pd.Categorical(df["class"], categories=["Class 1", "Class 2", "Class 3"])
df["instance"] = pd.Categorical(df["instance"], categories=["predict", "reject"])

p = (
    ggplot(df, aes(x="class", y="probability", fill="class"))
    + geom_col()
    + facet_wrap("~ instance", ncol=2)
    + labs(x="Class", y="Probability", title="Softmax outputs: predict vs reject")
    + theme_minimal()
    + theme(
        axis_title=element_text(size=12),
        strip_text=element_text(size=12),
        legend_position="none",
    )
)
p.save("confidence_reject_example.pdf", width=10, height=4, dpi=150)


import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_hline,
    geom_vline,
    annotate,
    labs,
    theme_minimal,
    theme,
    element_text,
    scale_color_manual,
    scale_y_continuous,
)

# Input range
x = np.linspace(0, 10, 300)
# P(C1|x): high on left, low on right (sigmoid in -x)
p_c1 = 1 / (1 + np.exp((x - 5) / 1.5))
p_c2 = 1 - p_c1

df = pd.DataFrame({
    "x": np.concatenate([x, x]),
    "p": np.concatenate([p_c1, p_c2]),
    "class": ["$p(C_1|x)$"] * len(x) + ["$p(C_2|x)$"] * len(x),
})

theta = 0.85
inv_sigmoid = lambda p: 5 + 1.5 * np.log(p / (1 - p))
x_at_theta_c1 = inv_sigmoid(theta)
x_at_theta_c2 = inv_sigmoid(1 - theta)
x_reject_lo = min(x_at_theta_c1, x_at_theta_c2)
x_reject_hi = max(x_at_theta_c1, x_at_theta_c2)

p_reject = (
    ggplot(df, aes(x="x", y="p", color="class"))
    + geom_line(size=1.2)
    + geom_hline(yintercept=theta, linetype="dashed", color="green", size=1)
    + geom_vline(xintercept=x_reject_lo, color="green", size=0.8)
    + geom_vline(xintercept=x_reject_hi, color="green", size=0.8)
    + annotate("text", x=(x_reject_lo + x_reject_hi) / 2, y=0.15, label="reject region", size=11)
    + annotate("segment", x=x_reject_lo, xend=x_reject_hi, y=0.08, yend=0.08, color="green", size=0.8)
    + labs(x="x", y="$p(C|x)$", title="Reject Option")
    + scale_color_manual(values=["#1f77b4", "#d62728"])
    + scale_y_continuous(limits=(0, 1))
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, color="red"),
        axis_title=element_text(size=12),
        legend_title=element_blank(),
        legend_position=(0.85, 0.7),
    )
)
p_reject.save("reject_option_diagram.pdf", width=7, height=4.5, dpi=150)


# ---------------------------------------------------------------------------
# Risk–coverage curve: coverage (x) vs selective risk or selective accuracy (y)
# As coverage decreases, selective accuracy increases (risk decreases).
# ---------------------------------------------------------------------------
coverage = np.linspace(0.15, 1.0, 50)
# Selective accuracy increases as coverage decreases (only predict when confident)
# At coverage=1: standard accuracy; as coverage -> 0: accuracy on retained set rises
standard_accuracy = 0.72
selective_accuracy = 1 - (1 - standard_accuracy) * (coverage ** 0.5)  # curve shape
selective_risk = 1 - selective_accuracy

df_rc = pd.DataFrame({
    "coverage": coverage,
    "selective_accuracy": selective_accuracy,
    "selective_risk": selective_risk,
})

from plotnine import geom_point, geom_line, annotate, scale_x_continuous

p_risk_cov = (
    ggplot(df_rc, aes(x="coverage", y="selective_accuracy"))
    + geom_line(size=1.2)
    + geom_point(size=2)
    + annotate(
        "point", x=1.0, y=standard_accuracy, size=4, color="red"
    )
    + annotate(
        "text",
        x=1.02,
        y=standard_accuracy,
        label="Standard accuracy",
        size=13,
    )
    + scale_x_continuous(limits=(0, 1.12))
    + labs(
        x="Coverage",
        y="Selective accuracy",
        title="Risk–coverage trade-off",
    )
    + theme_minimal()
    + theme(
        axis_title=element_text(size=12),
        plot_title=element_text(size=12),
    )
)
p_risk_cov.save("risk_coverage_curve.pdf", width=9, height=4.5, dpi=150)


# ---------------------------------------------------------------------------
# Distance-based vs confusion-based rejection: 2D feature space
# Two clusters (circles, crosses); red points far away (Case a); shaded overlap (Case b).
# ---------------------------------------------------------------------------
np.random.seed(42)
n1, n2 = 80, 80
# Cluster 1: circles
c1 = np.random.randn(n1, 2) * 0.6 + np.array([2, 2])
# Cluster 2: crosses
c2 = np.random.randn(n2, 2) * 0.6 + np.array([5, 5])
df_c1 = pd.DataFrame(c1, columns=["x", "y"])
df_c1["class"] = "C1"
df_c2 = pd.DataFrame(c2, columns=["x", "y"])
df_c2["class"] = "C2"
df_clusters = pd.concat([df_c1, df_c2], ignore_index=True)

# Red points: far from both clusters (distance-based reject)
red_points = np.array([[-2.2, -1.8], [7.5, 1.5], [1.2, 7.0], [-1.5, 5.5]])
df_red = pd.DataFrame(red_points, columns=["x", "y"])

# Confusion region: polygon in overlap between clusters (approx. 3 < x,y < 4.5)
confusion_region = np.array([
    [2.8, 3.0], [3.6, 2.7], [4.2, 3.2], [4.4, 4.0], [4.0, 4.5], [3.2, 4.3], [2.7, 3.8], [2.8, 3.0],
])
df_region = pd.DataFrame(confusion_region, columns=["x", "y"])

from plotnine import (
    geom_point,
    geom_polygon,
    scale_shape_manual,
    scale_color_manual,
    scale_x_continuous,
    scale_y_continuous,
)

p_dist_conf = (
    ggplot()
    + geom_polygon(data=df_region, mapping=aes(x="x", y="y"), fill="lightblue", alpha=0.5)
    + geom_point(data=df_clusters, mapping=aes(x="x", y="y", shape="class"), size=2.5, color="black", fill="white")
    + geom_point(data=df_red, mapping=aes(x="x", y="y"), size=4, color="red", shape="o")
    + annotate("text", x=-2.2, y=-2.4, label="(a)", size=11, fontweight="bold")
    + annotate("text", x=3.6, y=2.4, label="(b)", size=11, fontweight="bold")
    + labs(x="", y="", title="Distance-based vs confusion-based rejection")
    + scale_shape_manual(values=["o", "x"])
    + scale_x_continuous(limits=(-3.5, 8.5))
    + scale_y_continuous(limits=(-2.8, 8))
    + theme_minimal()
    + theme(
        plot_title=element_text(size=12),
        legend_position=(0.92, 0.85),
        legend_title=element_blank(),
    )
)
p_dist_conf.save("distance_vs_confusion_reject.pdf", width=6, height=5.5, dpi=150)


# ---------------------------------------------------------------------------
# System accuracy vs deferral rate: model-only, expert-only, deferral system
# Deferral system curve is above both for intermediate deferral rates.
# ---------------------------------------------------------------------------
deferral_rate = np.linspace(0, 1, 80)
model_acc = 0.72
expert_acc = 0.78
# Deferral system: blends model and expert, with extra gain when deferring the right cases (peak near middle)
bonus = 0.12 * 4 * deferral_rate * (1 - deferral_rate)  # max at rate=0.5
system_acc = model_acc * (1 - deferral_rate) + expert_acc * deferral_rate + bonus

df_model = pd.DataFrame({"deferral_rate": deferral_rate, "accuracy": np.full_like(deferral_rate, model_acc), "system": "Model only"})
df_expert = pd.DataFrame({"deferral_rate": deferral_rate, "accuracy": np.full_like(deferral_rate, expert_acc), "system": "Expert only"})
df_defer = pd.DataFrame({"deferral_rate": deferral_rate, "accuracy": system_acc, "system": "Deferral system"})
df_sys = pd.concat([df_model, df_expert, df_defer], ignore_index=True)

p_sys_acc = (
    ggplot(df_sys, aes(x="deferral_rate", y="accuracy", color="system"))
    + geom_line(size=1.2)
    + labs(x="Deferral rate", y="System accuracy", title="System accuracy vs deferral rate")
    + scale_x_continuous(limits=(0, 1))
    + scale_y_continuous(limits=(0.65, 0.95))
    + scale_color_manual(values=["#1f77b4", "#d62728", "#2ca02c"])
    + theme_minimal()
    + theme(
        axis_title=element_text(size=12),
        plot_title=element_text(size=12),
        legend_position=(0.55, 0.25),
        legend_title=element_blank(),
    )
)
p_sys_acc.save("system_accuracy_vs_deferral.pdf", width=6, height=4, dpi=150)