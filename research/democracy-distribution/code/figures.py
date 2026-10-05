"""Figures for the research note. Run from this directory: python figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from model import Econ
from markov import Markov

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "legend.frameon": False, "savefig.dpi": 200})

# ---------- Figure 1: surplus geometry ----------
e = Econ()
D = np.linspace(0.0, 2.2, 400)
lo, hi = np.sqrt(e.Dstar2(e.tauD)), np.sqrt(e.Dstar2(e.tauA))
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.plot(D, e.SM(D**2), color=C1, ls="-", label=r"median $S_M(\Delta^2)$")
e.z = 2.326; ax.plot(D, e.SE(D), color=C2, ls="--", label=r"top-1% elite $S_E(\Delta)$, $z=2.33$")
dh1 = e.Dhat()
e.z = 1.10; ax.plot(D, e.SE(D), color=C3, ls="-.", label=r"broad oligarchy $S_E(\Delta)$, $z=1.10$")
dh2 = e.Dhat()
ax.axhline(0, color=INK2, lw=0.8)
ax.axhline(-0.62, color=INK2, lw=0.8, ls=":")
ax.text(2.18, -0.60, r"$-\psi$", ha="right", va="bottom", color=INK2)
for v, lab in [(lo, r"$\Delta^*_D$"), (hi, r"$\Delta^*_A$")]:
    ax.axvline(v, color=INK2, lw=0.8, ls=":"); ax.text(v, 0.95, lab, ha="center", va="bottom", color=INK2)
ax.annotate("distributive branch\n(surplus falls with $\\Delta$)", xy=(1.05, -0.66), xytext=(1.15, -1.35), color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate("productive branch\n(surplus rises with $\\Delta$)", xy=(1.6, 0.10), xytext=(1.25, 0.55), color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlabel(r"inequality $\Delta$ (s.d. of log income)"); ax.set_ylabel("democratic surplus (dynastic log units)")
ax.set_ylim(-1.5, 1.1); ax.set_xlim(0, 2.2)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_surplus.pdf")); plt.close(fig)
print("Fig1: Dhat top1%%=%.3f, broad=%.3f" % (dh1, dh2))

# ---------- Figure 2: phase diagram ----------
mu = 0.35
S = np.linspace(0.25, 1.15, 300); PSI = np.linspace(0.2, 0.9, 300)
cls = np.zeros((PSI.size, S.size))
sbar = None
for j, s in enumerate(S):
    ee = Econ(s2=s**2)
    A_st = ee.SM(ee.Dstar2(ee.tauA)) < mu
    psic = -ee.SE(np.sqrt(ee.Dstar2(ee.tauD)))
    for i, psi in enumerate(PSI):
        Dc = psi > psic
        cls[i, j] = {(True, True): 0, (False, True): 1, (False, False): 2, (True, False): 3}[(A_st, Dc)]
fills = ["#d3e3f6", "#cdeee1", "#fbdccc", "#e4e3df"]   # light tints of slots 1,3,2 + neutral
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.pcolormesh(S, PSI, cls, cmap=ListedColormap(fills), vmin=-0.5, vmax=3.5, shading="auto", rasterized=True)
psic_curve = [-Econ(s2=s**2).SE(np.sqrt(Econ(s2=s**2).Dstar2(0.25))) for s in S]
ax.plot(S, psic_curve, color=INK2, lw=1.0)
sb = brentq(lambda s: Econ(s2=s**2).SM(Econ(s2=s**2).Dstar2(0.0)) - mu, 0.2, 1.5)
ax.axvline(sb, color=INK2, lw=1.0)
ax.text(0.33, 0.83, "both regimes stable\n(history decides)", color=INK, fontsize=8, va="top")
ax.text(sb+0.03, 0.83, "democratic convergence", color=INK, fontsize=8, va="top")
ax.text(sb+0.17, 0.33, "recurrent switching", color=INK, fontsize=8)
ax.text(0.27, 0.23, "autocratic\nconvergence", color=INK, fontsize=8)
pts = json.load(open(os.path.join(os.path.dirname(__file__), "phase_points.json")))
for pt in pts:
    agree = pt["closed"] == pt["markov"]
    ax.plot(pt["s"], pt["psi"], marker="o" if agree else "x", ms=4 if agree else 5, mfc="none" if agree else INK,
            mec=INK, lw=0, mew=0.8)
ax.plot([], [], "o", mfc="none", mec=INK, label="Markov equilibrium agrees"); ax.plot([], [], "x", color=INK, label="Markov equilibrium differs")
ax.legend(loc="lower right", fontsize=7.5)
ax.set_xlabel(r"s.d. of intergenerational income shocks $s$"); ax.set_ylabel(r"coup cost $\psi$")
ax.set_xlim(S[0], S[-1]); ax.set_ylim(PSI[0], PSI[-1]); ax.grid(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_phase.pdf")); plt.close(fig)
print("Fig2: sbar=%.3f" % sb)

# ---------- Figure 3: sample paths ----------
rng = np.random.default_rng(21)
cases = [("(a) $s=0.90$, $\\psi=0.70$", 0.90, 0.70, [(0, 0.55**2)]),
         ("(b) $s=0.90$, $\\psi=0.35$", 0.90, 0.35, [(1, 1.3**2)]),
         ("(c) $s=0.55$, $\\psi=0.70$", 0.55, 0.70, [(0, 0.6**2), (1, 0.9**2)])]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (title, s, psi, starts) in zip(axes, cases):
    ee = Econ(s2=s**2); m = Markov(ee, p=0.3, psi=psi, q=0.3, mu=mu, conc=False); m.solve()
    for k, (r0, D0) in enumerate(starts):
        sim = m.simulate(D0, r0, 18, rng)
        t = np.array([x[0] for x in sim]); dl = np.array([x[2] for x in sim]); rg = np.array([x[1] for x in sim])
        col = [C1, C2][k]; lab = ["starts in autocracy", "starts in democracy"][r0] if len(starts) > 1 else None
        ax.plot(t, dl, color=col, lw=1.4, label=lab)
        ax.plot(t[rg == 1], dl[rg == 1], "o", ms=3.5, color=col, mec="white", mew=0.5)
        ax.plot(t[rg == 0], dl[rg == 0], "s", ms=3.5, mfc="white", mec=col, mew=0.9)
    ax.axhline(np.sqrt(ee.Dstar2(0.25)), color=INK2, lw=0.7, ls=":"); ax.axhline(np.sqrt(ee.Dstar2(0.0)), color=INK2, lw=0.7, ls="--")
    ax.set_title(title, fontsize=8.5, loc="left", color=INK); ax.set_xlabel("generation")
    if len(starts) > 1: ax.legend(fontsize=6.5, loc="lower right")
axes[0].set_ylabel(r"inequality $\Delta_t$")
axes[0].plot([], [], "o", color=INK2, ms=3.5, label="democracy"); axes[0].plot([], [], "s", mfc="white", mec=INK2, ms=3.5, label="autocracy")
axes[0].plot([], [], ":", color=INK2, label=r"$\Delta^*_D$"); axes[0].plot([], [], "--", color=INK2, label=r"$\Delta^*_A$")
axes[0].legend(fontsize=6.5, loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_paths.pdf")); plt.close(fig)

# ---------- Figure 4: breakdown hazard by democratic age ----------
e = Econ(); psi, p = 0.62, 0.3
f = lambda x: e.SE(x) + psi; DC = brentq(f, 0.5, e.Dhat())
DD2, rD = e.Dstar2(e.tauD), e.rho(e.tauD)
m = Markov(e, p=p, psi=psi, q=0.0, mu=9, conc=False); m.solve()
rng = np.random.default_rng(31); D0s = rng.uniform(0.85, 1.6, 40000)
alive = np.ones(D0s.size, bool); D2 = D0s**2; hs = []
for k in range(5):
    o = m.outcomes(D2); fall = alive & (rng.random(D0s.size) < p) & o["coup"]
    hs.append(fall.sum()/alive.sum()); alive &= ~fall; D2 = rD**2*D2 + e.s2
T = np.array([0 if d <= DC else int(np.ceil(np.log((d**2-DD2)/(DC**2-DD2))/np.log(1/rD**2))) for d in D0s])
hc = []
for k in range(5):
    w = (1-p)**np.minimum(T, k); hc.append(p*w[T > k].sum()/w.sum())
fig, ax = plt.subplots(figsize=(4.4, 2.8))
ages = np.arange(5)
ax.bar(ages, hs, width=0.55, color=C1, label="simulated (Markov equilibrium)")
ax.plot(ages, hc, "o", color=C2, ms=5, label="closed form")
for a_, h_ in zip(ages, hs):
    ax.text(a_, h_ + 0.008, f"{h_:.2f}", ha="center", fontsize=7, color=INK)
ax.set_xlabel("democratic age (generations)"); ax.set_ylabel("breakdown hazard")
ax.legend(fontsize=7.5); ax.set_ylim(0, 0.34)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_hazard.pdf")); plt.close(fig)
print("Fig4: hazard sim", np.round(hs, 4), "closed", np.round(hc, 4), "Delta_C=%.3f" % DC)
