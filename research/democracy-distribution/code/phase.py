import numpy as np, json, time
from model import Econ
from markov import Markov
rng = np.random.default_rng(7)
mu, p, q = 0.35, 0.3, 0.3
def closed(s, psi):
    e = Econ(s2=s**2)
    Astable = e.SM(e.Dstar2(e.tauA)) < mu
    Dcons = e.SE(np.sqrt(e.Dstar2(e.tauD))) > -psi
    return {(True, True): 'both', (False, True): 'D', (False, False): 'switch', (True, False): 'A'}[(Astable, Dcons)]
def markov_class(s, psi):
    e = Econ(s2=s**2)
    m = Markov(e, p=p, psi=psi, q=q, mu=mu, conc=False); ok = m.solve()
    sh = []
    for r0, D0 in [(0, e.Dstar2(e.tauA)), (1, e.Dstar2(e.tauD))]:
        sims = [m.simulate(D0, r0, 160, rng) for _ in range(60)]
        sh.append(np.mean([x[1] for sim in sims for x in sim[80:]]))
    a, d = sh
    if a < 0.05 and d > 0.95: c = 'both'
    elif a > 0.95 and d > 0.95: c = 'D'
    elif a < 0.05 and d < 0.05: c = 'A'
    else: c = 'switch'
    return c, a, d, ok
t0 = time.time()
pts = []
for s in np.linspace(0.3, 1.1, 9):
    for psi in np.linspace(0.25, 0.85, 7):
        c, a, d, ok = markov_class(s, psi)
        pts.append(dict(s=float(s), psi=float(psi), closed=closed(s, psi), markov=c, shareD_fromA=float(a), shareD_fromD=float(d), converged=bool(ok)))
agree = np.mean([pt['closed'] == pt['markov'] for pt in pts])
print("time %.1fs, agreement closed vs markov: %.2f" % (time.time()-t0, agree))
for pt in pts:
    if pt['closed'] != pt['markov']: print(pt)
json.dump(pts, open('phase_points.json', 'w'))
