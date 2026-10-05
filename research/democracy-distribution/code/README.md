# Code for "Inequality and Political Regimes as a Joint Dynamic State"

Python 3 with numpy, scipy, sympy, matplotlib. Run each script from this directory.

| Script | What it checks |
|---|---|
| `model.py` | Closed forms: Pi, B, steady states, surpluses S_M and S_E, Delta-hat |
| `markov.py` | Markov equilibrium solver (coups, revolutions, optional concessions and voluntary handover) and simulator |
| `check_algebra.py` | Symbolic check of the aggregation lemma and value-function sums |
| `mc_check.py` | Monte Carlo population (400k dynasties) vs closed-form dynastic values |
| `explore_transitions.py` | Concessions on/off, threat frequency, revolution and coup costs |
| `phase.py` | Closed-form taxonomy vs Markov outcomes on a 9x7 grid; writes `phase_points.json` |
| `hazard.py` | Breakdown hazard by democratic age; beta comparative statics |
| `productive_branch_*.py` | Productive-branch experiments: cheap reversal, rare reversal, convergence diagnostics |
| `figures.py` | Produces the four figures in `../figures` (needs `phase_points.json`) |
