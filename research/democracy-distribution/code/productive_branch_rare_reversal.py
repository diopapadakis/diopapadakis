import numpy as np
from model import Econ
from markov import Markov
rng = np.random.default_rng(5)
for name, e in {'productive (z=1.10)': Econ(tauA=-0.25, tauD=0.35, s2=0.47, z=1.10),
                'distributive (z=2.326)': Econ(tauA=-0.25, tauD=0.35, s2=0.47, z=2.326),
                'distributive (z=1.645)': Econ(tauA=-0.25, tauD=0.35, s2=0.47, z=1.645)}.items():
    lo, hi = np.sqrt(e.Dstar2(e.tauD)), np.sqrt(e.Dstar2(e.tauA))
    print(f"\n{name}: D*_D={lo:.3f} D*_A={hi:.3f} Dhat={e.Dhat():.3f}  SE(D*_D)={e.SE(lo):.3f} SE(D*_A)={e.SE(hi):.3f}")
    for p in [0.1, 0.25]:
        for cost in [0.0, 0.02]:
            m = Markov(e, p=p, psi=cost, q=0.0, mu=1.0, conc=False, psiV=cost); ok = m.solve()
            th = m.thresholds()
            res = []
            for r0, D0 in [(0, 0.3**2), (0, 1.3**2), (1, 0.6**2), (1, 1.3**2)]:
                sims = [m.simulate(D0, r0, 120, rng) for _ in range(100)]
                shareD = np.mean([x[1] for s in sims for x in s[60:]])
                sw = np.mean([sum(1 for x in s if x[3] != '' ) for s in sims])
                res.append(f"{'AD'[r0]}{np.sqrt(D0):.1f}->D share {shareD:.2f} sw {sw:.1f}")
            print(f"  p={p} cost={cost}: coup{th['coup']} vol{th['vol']} | " + "; ".join(res))
