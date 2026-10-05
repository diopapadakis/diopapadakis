import sympy as sp
m, x, tau, D2, d = sp.symbols('m x tau Delta2 d', real=True)
# Budget: ytilde^tau * E[h^(1-tau)] = (1-kappa) E[h],  ln h ~ N(m, D2), d = ln(1-kappa)
lnyt = sp.symbols('lnyt')
lhs = tau*lnyt + (1-tau)*m + (1-tau)**2*D2/2
rhs = d + m + D2/2
sol = sp.solve(sp.Eq(lhs, rhs), lnyt)[0]
print("ln ytilde =", sp.simplify(sol - (m + d/tau + (2-tau)*D2/2)))
lnyhat = (1-tau)*(m+x) + tau*sol
print("ln yhat - [d + m + (1-tau)x + tau(2-tau)D2/2] =", sp.simplify(lnyhat - (d + m + (1-tau)*x + tau*(2-tau)*D2/2)))
# value-function constants
dl, al, be, rho, s2 = sp.symbols('delta alpha beta rho s2', positive=True)
a = al+be
omega = 1 + dl*be/(1-dl*a)
print("omega simplifies to", sp.simplify(omega - (1-dl*al)/(1-dl*a)))
# sum_k delta^k Delta2_{t+k}
k = sp.symbols('k', integer=True, nonnegative=True)
Dk = rho**(2*k)*D2 + s2*(1-rho**(2*k))/(1-rho**2)
S = sp.summation(dl**k*Dk, (k, 0, sp.oo))
S = sp.simplify(sp.piecewise_fold(S).args[0][0]) if isinstance(S, sp.Piecewise) else S
print("sum check:", sp.simplify(S - (D2/(1-dl*rho**2) + dl*s2/((1-dl)*(1-dl*rho**2)))))
