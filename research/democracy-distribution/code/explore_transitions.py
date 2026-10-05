import numpy as np
from model import Econ
from markov import Markov
rng = np.random.default_rng(2)
e = Econ()
def longrun(m, T=300, n=150):
    fr = []
    for r0 in [0, 1]:
        sims = [m.simulate(e.Dstar2(e.tauA) if r0==0 else e.Dstar2(e.tauD), r0, T, rng) for _ in range(n)]
        fr.append(np.mean([x[1] for sim in sims for x in sim[150:]]))
        nsw = np.mean([sum(1 for x in sim if x[3] in ('coup','democratization(threat)')) for sim in sims])
        fr.append(nsw)
    return fr
for conc in [True, False]:
    for q in [0.1, 0.3]:
        for mu in [0.05, 0.10, 0.20]:
            for psi in [0.45, 0.62]:
                m = Markov(e, p=0.3, psi=psi, q=q, mu=mu, conc=conc); m.solve()
                th = m.thresholds(); fr = longrun(m)
                print(f"conc={conc} q={q} mu={mu} psi={psi}: coup{th['coup']} dem{th['thr_dem']} | shareD fromA {fr[0]:.2f} (switches {fr[1]:.1f}), fromD {fr[2]:.2f} (switches {fr[3]:.1f})")
