import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_abline,
    theme_minimal, labs, scale_color_manual, theme,
    element_text, element_rect
)

# -- Generate a classification dataset
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -- Fit several models with different calibration profiles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM (Platt)": SVC(probability=True, random_state=42),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
    ),
}

colors = {
    "Logistic Regression": "#1565C0",
    "Random Forest": "#2E7D32",
    "Gradient Boosting": "#E65100",
    "SVM (Platt)": "#6A1B9A",
    "Neural Network": "#C62828",
}

# -- Compute calibration curves
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    for pt, pp in zip(prob_true, prob_pred):
        rows.append({"Model": name, "Predicted Probability": pp, "Observed Frequency": pt})

df = pd.DataFrame(rows)

# -- Plot
p = (
    ggplot(df, aes(x="Predicted Probability", y="Observed Frequency", color="Model"))
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray", size=0.8)
    + geom_line(size=1)
    + geom_point(size=2.5)
    + scale_color_manual(values=colors)
    + theme_minimal()
    + theme(
        legend_position="right",
        legend_title=element_text(weight="bold"),
        plot_title=element_text(size=14, weight="bold"),
        plot_subtitle=element_text(size=10),
    )
    + labs(
        x="Mean Predicted Probability",
        y="Observed Frequency (Fraction of Positives)",
        title="Reliability Diagram",
        subtitle="Dashed diagonal = perfect calibration",
        color="Model",
    )
)

p.save("reliability_diagram.pdf", width=9, height=6, dpi=200)
print(p)


# =====================================================
# ECE before and after Platt Scaling
# =====================================================

from sklearn.calibration import CalibratedClassifierCV


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece


# -- Load LFW pairs dataset (face verification: same person or not)
from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA

print("Loading LFW pairs dataset...")
lfw = fetch_lfw_pairs(subset="train", resize=0.5)
X_lfw = lfw.data
y_lfw = lfw.target  # 0 = different person, 1 = same person

# Reduce dimensionality (raw pixels are high-dimensional)
pca = PCA(n_components=50, random_state=42)
X_lfw = pca.fit_transform(X_lfw)

# -- Split data: train / calibration / test
X_train2, X_temp, y_train2, y_temp = train_test_split(
    X_lfw, y_lfw, test_size=0.4, random_state=42
)
X_cal, X_test2, y_cal, y_test2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# -- Train a Naive Bayes classifier (often poorly calibrated)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train2, y_train2)

# -- Before calibration
y_prob_before = nb.predict_proba(X_test2)[:, 1]
ece_before = compute_ece(y_test2, y_prob_before)

# -- Apply Platt Scaling (sigmoid method) using calibration set
nb_calibrated = CalibratedClassifierCV(nb, method="sigmoid", cv=5)
nb_calibrated.fit(X_cal, y_cal)

# -- After calibration
y_prob_after = nb_calibrated.predict_proba(X_test2)[:, 1]
ece_after = compute_ece(y_test2, y_prob_after)

print(f"ECE before Platt Scaling: {ece_before:.4f}")
print(f"ECE after Platt Scaling:  {ece_after:.4f}")

# -- Compute calibration curves for both
prob_true_before, prob_pred_before = calibration_curve(
    y_test2, y_prob_before, n_bins=10, strategy="uniform"
)
prob_true_after, prob_pred_after = calibration_curve(
    y_test2, y_prob_after, n_bins=10, strategy="uniform"
)

# -- Build dataframe for plotting
rows_cal = []
for pt, pp in zip(prob_true_before, prob_pred_before):
    rows_cal.append({
        "Predicted Probability": pp,
        "Observed Frequency": pt,
        "Model": f"Before (ECE = {ece_before:.3f})",
    })
for pt, pp in zip(prob_true_after, prob_pred_after):
    rows_cal.append({
        "Predicted Probability": pp,
        "Observed Frequency": pt,
        "Model": f"After Platt Scaling (ECE = {ece_after:.3f})",
    })

df_cal = pd.DataFrame(rows_cal)

colors_cal = {
    f"Before (ECE = {ece_before:.3f})": "#C62828",
    f"After Platt Scaling (ECE = {ece_after:.3f})": "#1565C0",
}

# -- Plot
p_cal = (
    ggplot(df_cal, aes(x="Predicted Probability", y="Observed Frequency", color="Model"))
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray", size=0.8)
    + geom_line(size=1.1)
    + geom_point(size=3)
    + scale_color_manual(values=colors_cal)
    + theme_minimal()
    + theme(
        legend_position="bottom",
        legend_title=element_text(weight="bold"),
        plot_title=element_text(size=14, weight="bold"),
        plot_subtitle=element_text(size=10),
    )
    + labs(
        x="Mean Predicted Probability",
        y="Observed Frequency (Fraction of Positives)",
        title="Calibration: Before vs. After Platt Scaling",
        subtitle="Naive Bayes on LFW face verification",
        color="",
    )
)

p_cal.save("calibration_before_after_real.pdf", width=12, height=5, dpi=200)


