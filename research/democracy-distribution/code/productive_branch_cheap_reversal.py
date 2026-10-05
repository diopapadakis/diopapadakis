import numpy as np
from model import Econ
from markov import Markov
rng = np.random.default_rng(4)
# Elite controls both directions: voluntary democratization (every period, cost psiV) and coups (p=1, cost psi); no revolutions (q=0).
cases = {'productive branch (z=1.10)': Econ(tauA=-0.25, tauD=0.35, s2=0.47, z=1.10),
         'distributive branch (z=2.326)': Econ(tauA=-0.25, tauD=0.35, s2=0.47, z=2.326)}
for name, e in cases.items():
    lo, hi = np.sqrt(e.Dstar2(e.tauD)), np.sqrt(e.Dstar2(e.tauA))
    print(f"\n{name}: D*_D={lo:.3f} D*_A={hi:.3f} Dhat={e.Dhat():.3f}")
    for cost in [0.005, 0.02]:
        m = Markov(e, p=1.0, psi=cost, q=0.0, mu=1.0, conc=False, psiV=cost); ok = m.solve()
        th = m.thresholds()
        for r0, D0 in [(0, e.Dstar2(e.tauD)), (1, e.Dstar2(e.tauA)), (0, 0.3), (1, 1.6)]:
            sim = m.simulate(D0, r0, 40, rng)
            regs = ''.join('D' if x[1] else 'A' for x in sim)
            print(f"  cost={cost} start r0={'AD'[r0]} Delta0={np.sqrt(D0):.2f}: coup{th['coup']} vol{th['vol']} path {regs[:30]}  Delta_end={sim[-1][2]:.3f}")
