import numpy as np
from scipy.optimize import brentq
from model import Econ
from markov import Markov
rng = np.random.default_rng(11)
e = Econ(); psi, p = 0.62, 0.3
# closed-form coup zone (A absorbing)
f = lambda D: e.SE(D) + psi
DC = brentq(f, 0.5, e.Dhat()); DCup = brentq(f, e.Dhat(), 4.0)
DD2 = e.Dstar2(e.tauD); rD = e.rho(e.tauD)
print(f"coup zone closed form: ({DC:.3f}, {DCup:.3f}); Delta*_D={np.sqrt(DD2):.3f}")
def T(D0):  # periods until Delta <= DC under democracy
    if D0 <= DC: return 0
    return int(np.ceil(np.log((D0**2-DD2)/(DC**2-DD2))/np.log(1/rD**2)))
m = Markov(e, p=p, psi=psi, q=0.0, mu=9, conc=False); print("converged:", m.solve())
print("Markov thresholds:", m.thresholds())
D0s = rng.uniform(0.85, 1.6, 20000)
ages = 8
alive = np.ones(D0s.size, bool); haz_sim = []
D2 = D0s**2; r = np.ones(D0s.size, int)
for k in range(ages):
    o = m.outcomes(D2)
    opp = rng.random(D0s.size) < p
    fall = alive & opp & o['coup']
    haz_sim.append(fall.sum()/alive.sum())
    alive &= ~fall
    D2 = np.where(alive, e.rho(e.tauD)**2*D2 + e.s2, D2)
Ts = np.array([T(d) for d in D0s])
haz_cf = []
for k in range(ages):
    inzone = (Ts > k); w = np.where(inzone, (1-p)**k, 1.0)
    haz_cf.append(p*w[inzone].sum()/w.sum())
print("age  hazard_sim  hazard_closedform")
for k in range(ages): print(k, round(haz_sim[k], 4), round(haz_cf[k], 4))
np.save('hazard.npy', np.array([haz_sim, haz_cf]))
# beta comparative statics holding rho_A fixed (tauA=0 => alpha = 0.65 - beta)
print("\nbeta  rhoD   D*_D   stake(B_A-B_D)  Pi_D   psi_c=-SE(D*_D)  T(1.5)")
for be in [0.25, 0.35, 0.45, 0.55]:
    ee = Econ(al=0.65-be, be=be)
    lo = np.sqrt(ee.Dstar2(ee.tauD))
    psic = -ee.SE(lo)
    fC = lambda D: ee.SE(D) + 0.62
    try:
        DCb = brentq(fC, 0.3, ee.Dhat()); 
        Tb = np.log((1.5**2-lo**2)/(DCb**2-lo**2))/np.log(1/ee.rho(ee.tauD)**2) if DCb > lo else np.inf
    except ValueError: DCb, Tb = np.nan, np.nan
    print(f"{be:.2f}  {ee.rho(ee.tauD):.3f}  {lo:.3f}  {ee.B(ee.tauA)-ee.B(ee.tauD):.3f}  {ee.Pi(ee.tauD):.3f}  {psic:.3f}  DC={DCb:.3f} T={Tb:.2f}")
