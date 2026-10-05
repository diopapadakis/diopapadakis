import numpy as np
from model import Econ
from markov import Markov
e = Econ(tauA=-0.25, tauD=0.35, s2=0.47, z=1.10)
for damp, n in [(0.5, 700), (0.9, 700), (0.97, 1500)]:
    m = Markov(e, p=0.1, psi=0.02, q=0.0, mu=1.0, conc=False, psiV=0.02, grid=(0.02, 3.5, n))
    ok = m.solve(damp=damp, maxit=20000)
    # check whether the decision rule oscillates: record vol/coup masks over extra iterations
    masks = []
    for k in range(6):
        m.solve(damp=damp, maxit=50)
        o = m.outcomes(m.G); masks.append((o['vol'].sum(), o['coup'].sum()))
    print(f"damp={damp} n={n} converged={ok} (#vol states, #coup states) over iterations: {masks}")
