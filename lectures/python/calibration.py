import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_abline,
    theme_minimal, labs, scale_color_manual, theme,
    element_text,
)


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


# -- Generate dataset
X, y = make_classification(
    n_samples=8000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42,
)

# -- Three-way split: train / calibration / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# -- Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# -- Uncalibrated predictions
y_prob_uncal = nb.predict_proba(X_test)[:, 1]
ece_uncal = compute_ece(y_test, y_prob_uncal)

# -- Platt Scaling (sigmoid)
nb_platt = CalibratedClassifierCV(nb, method="sigmoid", cv=5)
nb_platt.fit(X_cal, y_cal)
y_prob_platt = nb_platt.predict_proba(X_test)[:, 1]
ece_platt = compute_ece(y_test, y_prob_platt)

# -- Isotonic Regression
nb_isotonic = CalibratedClassifierCV(nb, method="isotonic", cv=5)
nb_isotonic.fit(X_cal, y_cal)
y_prob_isotonic = nb_isotonic.predict_proba(X_test)[:, 1]
ece_isotonic = compute_ece(y_test, y_prob_isotonic)

# -- Temperature Scaling (manual implementation)
# Temperature scaling: find T that minimizes NLL on calibration set
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def temperature_nll(T, probs, y_true):
    """Negative log-likelihood with temperature scaling."""
    # Clip probabilities to avoid log(0) and logit issues
    probs_clipped = np.clip(probs, 1e-8, 1 - 1e-8)
    # Convert to logits, scale by temperature, convert back
    logits = logit(probs_clipped)
    scaled_probs = expit(logits / T)
    # Negative log-likelihood
    nll = -np.mean(y_true * np.log(scaled_probs) + (1 - y_true) * np.log(1 - scaled_probs))
    return nll


# Get uncalibrated probabilities on calibration set
y_prob_cal_uncal = nb.predict_proba(X_cal)[:, 1]

# Optimize temperature on calibration set
result = minimize_scalar(
    temperature_nll,
    bounds=(0.1, 10.0),
    args=(y_prob_cal_uncal, y_cal),
    method="bounded",
)
T_opt = result.x
print(f"Optimal temperature: {T_opt:.3f}")

# Apply temperature scaling to test set
y_prob_uncal_clipped = np.clip(y_prob_uncal, 1e-8, 1 - 1e-8)
y_prob_temp = expit(logit(y_prob_uncal_clipped) / T_opt)
ece_temp = compute_ece(y_test, y_prob_temp)

# -- Print ECE values
print(f"ECE Uncalibrated:        {ece_uncal:.4f}")
print(f"ECE Platt Scaling:       {ece_platt:.4f}")
print(f"ECE Isotonic Regression: {ece_isotonic:.4f}")
print(f"ECE Temperature Scaling: {ece_temp:.4f}")

# -- Build dataframe for plotting
methods = {
    f"Uncalibrated (ECE={ece_uncal:.3f})": y_prob_uncal,
    f"Platt Scaling (ECE={ece_platt:.3f})": y_prob_platt,
    f"Isotonic Regression (ECE={ece_isotonic:.3f})": y_prob_isotonic,
    f"Temperature Scaling (ECE={ece_temp:.3f})": y_prob_temp,
}

rows = []
for name, y_prob in methods.items():
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    for pt, pp in zip(prob_true, prob_pred):
        rows.append({
            "Predicted Probability": pp,
            "Observed Frequency": pt,
            "Method": name,
        })

df = pd.DataFrame(rows)

colors = {
    f"Uncalibrated (ECE={ece_uncal:.3f})": "#C62828",
    f"Platt Scaling (ECE={ece_platt:.3f})": "#1565C0",
    f"Isotonic Regression (ECE={ece_isotonic:.3f})": "#2E7D32",
    f"Temperature Scaling (ECE={ece_temp:.3f})": "#E65100",
}

# -- Plot
p = (
    ggplot(df, aes(x="Predicted Probability", y="Observed Frequency", color="Method"))
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray", size=0.8)
    + geom_line(size=1.1)
    + geom_point(size=2.5)
    + scale_color_manual(values=colors)
    + theme_minimal()
    + theme(
        legend_position="right",
        legend_direction="vertical",
        legend_title=element_text(weight="bold"),
        plot_title=element_text(size=14, weight="bold"),
        plot_subtitle=element_text(size=10),
    )
    + labs(
        x="Mean Predicted Probability",
        y="Observed Frequency (Fraction of Positives)",
        title="Calibration: Naive Bayes Before and After Calibration",
        subtitle="Comparison of Platt Scaling, Isotonic Regression, and Temperature Scaling",
        color="",
    )
)

p.save("calibration_methods.pdf", width=12, height=5, dpi=200)

