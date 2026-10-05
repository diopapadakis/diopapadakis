import numpy as np
rng = np.random.default_rng(0)
al, be, dl, s2 = 0.2, 0.45, 0.5, 0.47
a = al+be
tauA, tauD, theta = 0.0, 0.25, 1.0
dfun = lambda t: -theta*t**2/2
g1 = lambda t: t*(2-t)/2
rho = lambda t: al + be*(1-t)
N, T = 400_000, 60
path = ['D']*3 + ['A']*(T-3)          # a regime path with a switch
tau = {'A': tauA, 'D': tauD}
D2_0, m0 = 0.81, 0.0
lnh = m0 + np.sqrt(D2_0)*rng.standard_normal(N)
x0 = lnh - lnh.mean()
V = np.zeros(N)
for t in range(T):
    tt = tau[path[t]]
    mt, D2t = lnh.mean(), lnh.var()
    # exact budget-balanced scheme on the simulated population (no lognormal assumption here)
    h = np.exp(lnh)
    lny_tilde = (np.log((1-0)*np.exp(dfun(tt))*h.mean()) - np.log(np.mean(h**(1-tt))))/tt if tt != 0 else 0.0
    lnyhat = (1-tt)*lnh + tt*lny_tilde if tt != 0 else lnh + dfun(tt)
    V += dl**t * lnyhat                      # ln c up to the constant ln(1-sigma)
    lnh = al*lnh + be*lnyhat + np.sqrt(s2)*rng.standard_normal(N)   # kappa0 = 0, sigma folded in
# closed form: V(x) = m0/(1-dl a) + sum_k dl^k omega-terms ... compute via recursion along the path
om = 1 + dl*be/(1-dl*a)
D2 = D2_0; W = 0.0; B = 0.0; prod = 1.0
for t in range(T):
    tt = tau[path[t]]
    W += dl**t * om*(dfun(tt) + g1(tt)*D2)
    B += dl**t * (1-tt)*prod
    prod *= rho(tt)
    D2 = rho(tt)**2*D2 + s2
slope, icpt = np.polyfit(x0, V, 1)
print(f"slope MC {slope:.4f}  closed-form B {B:.4f}")
print(f"intercept MC {icpt:.4f}  closed-form W {W:.4f}  (m0=0, truncation at T={T})")
