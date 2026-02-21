import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_segment,
    theme_minimal, labs, theme, element_text,
    element_blank, annotate, scale_x_continuous, scale_y_continuous,
)

# -- Concave utility curve (square root)
x = np.linspace(0, 2000, 500)
y = np.sqrt(x)

df = pd.DataFrame({"reward": x, "utility": y})

# -- Reference line at x=1000
u_at_1000 = np.sqrt(1000)

p = (
    ggplot(df, aes(x="reward", y="utility"))
    + geom_line(size=1.2, color="black")
    + geom_segment(
        aes(x=1000, xend=1000, y=0, yend=u_at_1000),
        color="gray", linetype="dashed", size=0.5,
    )
    + geom_segment(
        aes(x=0, xend=1000, y=u_at_1000, yend=u_at_1000),
        color="gray", linetype="dashed", size=0.5,
    )
    + scale_x_continuous(breaks=[0, 500, 1000, 1500, 2000])
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, weight="bold"),
        axis_title=element_text(size=12),
        panel_grid_minor=element_blank(),
    )
    + labs(
        x="€ reward",
        y="U",
    )
)

p.save("utility_curve.pdf", width=10, height=5, dpi=200)


#

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line,
    theme_minimal, labs, theme, element_text,
    element_blank, scale_color_manual, scale_linetype_manual,
)

# -- Three risk attitudes
x = np.linspace(1, 2000, 500)

rows = []
for xi in x:
    rows.append({"reward": xi, "U": np.sqrt(xi), "Risk attitude": "Risk-averse (concave)"})
    rows.append({"reward": xi, "U": xi / np.sqrt(2000), "Risk attitude": "Risk-neutral (linear)"})
    rows.append({"reward": xi, "U": (xi ** 2) / (2000 ** 2) * np.sqrt(2000), "Risk attitude": "Risk-seeking (convex)"})

df = pd.DataFrame(rows)

colors = {
    "Risk-averse (concave)": "#1565C0",
    "Risk-neutral (linear)": "#555555",
    "Risk-seeking (convex)": "#C62828",
}

linetypes = {
    "Risk-averse (concave)": "solid",
    "Risk-neutral (linear)": "dashed",
    "Risk-seeking (convex)": "dotted",
}

p = (
    ggplot(df, aes(x="reward", y="U", color="Risk attitude", linetype="Risk attitude"))
    + geom_line(size=1.2)
    + scale_color_manual(values=colors)
    + scale_linetype_manual(values=linetypes)
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, weight="bold"),
        axis_title=element_text(size=12),
        panel_grid_minor=element_blank(),
        legend_position="right",
        legend_direction="vertical",
        legend_title=element_blank(),
    )
    + labs(
        x="€ reward",
        y="U",
        title="Utility Functions and Risk Attitudes",
    )
)

p.save("utility_curve2.pdf", width=10, height=5, dpi=200)
print(p)



import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_segment, geom_point,
    annotate, theme_minimal, labs, theme, element_text,
    element_blank, scale_x_continuous, scale_y_continuous,arrow,
)



# -- Concave utility curve (square root)
x = np.linspace(0, 2000, 500)
y = np.sqrt(x)

df = pd.DataFrame({"reward": x, "utility": y})

# -- Gamble: 50/50 chance of 0 or 2000
x_lo, x_hi = 200, 1800
u_lo, u_hi = np.sqrt(x_lo), np.sqrt(x_hi)

# Expected monetary value
emv = 0.5 * x_lo + 0.5 * x_hi  # = 1000
u_emv = np.sqrt(emv)

# Expected utility of the gamble
eu_gamble = 0.5 * u_lo + 0.5 * u_hi

# Certainty equivalent: the x where U(x) = EU(gamble)
ce = eu_gamble ** 2  # since U = sqrt(x), x = U^2

p = (
    ggplot(df, aes(x="reward", y="utility"))
    + geom_line(size=1.2, color="black")

    # Chord connecting the two gamble outcomes
    + geom_segment(aes(x=x_lo, xend=x_hi, y=u_lo, yend=u_hi),
                   color="#C62828", size=0.7, linetype="dashed")

    # Points at gamble outcomes
    + geom_point(aes(x=x_lo, y=u_lo), color="#C62828", size=3)
    + geom_point(aes(x=x_hi, y=u_hi), color="#C62828", size=3)

    # EU of the gamble (on the chord)
    + geom_point(aes(x=emv, y=eu_gamble), color="#C62828", size=3)
    + geom_segment(aes(x=emv, xend=emv, y=0, yend=eu_gamble),
                   color="#C62828", size=0.5, linetype="dotted")

    # U(EMV) -- utility of the expected value (on the curve)
    + geom_point(aes(x=emv, y=u_emv), color="#1565C0", size=3)
    + geom_segment(aes(x=emv, xend=emv, y=0, yend=u_emv),
                   color="#1565C0", size=0.5, linetype="dotted")

    # Certainty equivalent (on the curve, same height as EU)
    + geom_point(aes(x=ce, y=eu_gamble), color="#2E7D32", size=3)
    + geom_segment(aes(x=ce, xend=ce, y=0, yend=eu_gamble),
                   color="#2E7D32", size=0.5, linetype="dotted")
    + geom_segment(aes(x=0, xend=ce, y=eu_gamble, yend=eu_gamble),
                   color="#2E7D32", size=0.5, linetype="dotted")

    # Risk premium bracket (horizontal arrow between CE and EMV)
    + annotate("segment", x=ce, xend=emv, y=eu_gamble - 1.5,
               yend=eu_gamble - 1.5, color="#E65100", size=0.8,
               arrow=arrow(ends="both", length=0.1))

    # Labels
    + annotate("text", x=x_lo - 50, y=u_lo + 1.5, label="U(200)",
               size=8, ha="right", color="#C62828")
    + annotate("text", x=x_hi + 50, y=u_hi + 1.5, label="U(1800)",
               size=8, ha="left", color="#C62828")
    + annotate("text", x=emv + 50, y=eu_gamble - 1.5,
               label="EU(gamble)", size=8, ha="left", color="#C62828")
    + annotate("text", x=emv + 50, y=u_emv + 1.5,
               label="U(EMV)", size=8, ha="left", color="#1565C0")
    + annotate("text", x=ce - 50, y=eu_gamble + 1.5,
               label="CE", size=9, ha="right", fontweight="bold", color="#2E7D32")
    + annotate("text", x=(ce + emv) / 2, y=eu_gamble - 3.5,
               label="Risk\npremium", size=8, ha="center", color="#E65100")

    + scale_x_continuous(breaks=[0, 200, 500, int(round(ce)), 1000, 1500, 1800, 2000])
    + theme_minimal()
    + theme(
        axis_title=element_text(size=12),
        panel_grid_minor=element_blank(),
    )
    + labs(x="€ reward", y="U")
)

p.save("certainty_equivalent.pdf", width=9, height=6, dpi=200)
print(p)