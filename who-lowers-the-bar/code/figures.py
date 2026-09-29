"""Generates the figures of the draft (PDF, vector) into ../figures/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 150,
})
INK, ACCENT, GREY = "#1f2d3d", "#c0392b", "#7f8c8d"

M = Model(q=0.6, s=1.0, s_th2=1.0, s_eps2=1.0)
gs = np.linspace(-2, 2, 81)

# ---------------- Figure 1: threshold and bar-raising; ex ante vs ex post ----------------
dst = np.array([M.dstar(g) for g in gs])
Esil = np.array([M.E_d_silent(t) for t in dst])
g0 = M.g0()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
a = ax[0]
a.plot(gs, dst, color=INK, lw=1.8, label=r"disclosure threshold $d^*(g)$")
a.plot(gs, Esil, color=ACCENT, lw=1.5, ls="--", label=r"$\mathbb{E}[d\mid\varnothing]$ (context presumed after silence)")
a.axvspan(gs[0], g0, color=ACCENT, alpha=0.07)
a.axvline(g0, color=GREY, lw=0.8, ls=":")
a.text(-1.95, 0.6, "marginal discloser\nraises the bar", fontsize=8, color=ACCENT, va="center")
a.text(0.3, -2.25, "marginal concealer forgoes\nlowering the bar", fontsize=8, color=INK)
a.set_xlabel(r"lead $g=m-c$  (underdog $\leftarrow$ $\rightarrow$ favorite)")
a.set_ylabel(r"context $d$ (higher = harder)")
a.legend(loc="upper left", fontsize=7.5)
a.set_ylim(-2.6, 2.6)
a.set_title("(a) Disclosure threshold", fontsize=10)

b = ax[1]
pa = np.array([M.p_disclose_exante(g) for g in gs])
pp = M.p_disclose_expost()
b.plot(gs, pa, color=INK, lw=1.8, label="before the performance")
b.axhline(pp, color=ACCENT, lw=1.5, ls="--", label="after the performance")
b.set_ylim(0, 1.02)
b.set_xlabel(r"lead $g$")
b.set_ylabel("share of informed agents disclosing")
b.legend(loc="lower left", fontsize=8)
b.set_title("(b) Before vs. after the performance", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_threshold.pdf"))
plt.close(fig)

# ---------------- Figure 2: regimes ----------------
gs2 = np.linspace(-2, 2, 41)
res = {r: np.array([M.regime(g, r) for g in gs2]) for r in ["voluntary", "mandatory", "banned"]}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
styles = {"mandatory": (INK, "-"), "voluntary": (ACCENT, "--"), "banned": (GREY, ":")}
labels = {"mandatory": "mandatory", "voluntary": "voluntary (equilibrium)", "banned": "never disclosed"}
for r in ["mandatory", "voluntary", "banned"]:
    ax[0].plot(gs2, res[r][:, 0] - res["mandatory"][:, 0], color=styles[r][0], ls=styles[r][1], lw=1.6, label=labels[r])
    ax[1].plot(gs2, res[r][:, 1], color=styles[r][0], ls=styles[r][1], lw=1.6, label=labels[r])
ax[0].axhline(0, color=GREY, lw=0.6)
ax[0].set_xlabel(r"lead $g$")
ax[0].set_ylabel("hiring probability,\nrelative to mandatory")
ax[0].set_title("(a) Gains from mandatory disclosure", fontsize=10)
ax[0].legend(fontsize=7.5, loc="lower right")
ax[1].set_xlabel(r"lead $g$")
ax[1].set_ylabel("expected residual variance of talent")
ax[1].set_title("(b) What the evaluator learns", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_regimes.pdf"))
plt.close(fig)

# ---------------- Figure 3: smooth payoffs ----------------
gs3 = np.linspace(-1.5, 1.5, 13)
fig, a = plt.subplots(1, 1, figsize=(4.2, 3.0))
a.plot(gs3, [M.dstar(g) for g in gs3], color=INK, lw=1.8, label=r"hiring threshold ($\omega\to 0$, closed form)")
for om, st in [(0.3, "--"), (1.0, "-."), (3.0, ":")]:
    a.plot(gs3, [M.dstar_smooth(g, om) for g in gs3], color=ACCENT, ls=st, lw=1.3, label=rf"$\omega={om}$")
a.plot(gs3, [M.dstar_smooth(g, np.inf) for g in gs3], color=GREY, lw=1.3, label=r"linear payoff ($\omega\to\infty$)")
a.set_xlabel(r"lead $g$")
a.set_ylabel(r"disclosure threshold")
a.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_smooth.pdf"))
plt.close(fig)

# numbers quoted in the text
print("beta(q)", round(M.beta, 4), "g0", round(g0, 4), "ex post share", round(pp, 4))
for g in [-1.0, 0.0, 1.0]:
    print("g", g, "d*", round(M.dstar(g), 3), "E[d|silent]", round(M.E_d_silent(M.dstar(g)), 3),
          "ex ante share", round(M.p_disclose_exante(g), 3),
          {r: tuple(np.round(M.regime(g, r), 3)) for r in ["voluntary", "mandatory", "banned"]})
