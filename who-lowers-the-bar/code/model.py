"""
Numerical companion to "Who Lowers the Bar? Expectation Management with a Bayesian Audience".

Model (baseline, Section 2 of the draft):
  talent     theta ~ N(m, s_th^2)            unknown to everyone
  context    d     ~ N(0, s^2)               difficulty; agent observes it w.p. q (verifiable)
  performance y = theta - d + eps, eps ~ N(0, s_eps^2)
  evaluator hires iff E[theta | y, message] >= c;   lead (standing) g = m - c.
Write z = y - m, V = s_th^2 + s_eps^2, k = s_th^2 / V.

Everything is computed on a grid in log-space; closed forms from the draft are checked against it.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import logsumexp


def beta_dye(q):
    """Unique root of b = -q phi(b) / (1 - q + q Phi(b)) (the Dye / Jung-Kwon constant)."""
    f = lambda b: b + q * norm.pdf(b) / (1 - q + q * norm.cdf(b))
    return brentq(f, -40, 5)


class Model:
    def __init__(self, q=0.6, s=1.0, s_th2=1.0, s_eps2=1.0, n=4001, width=9.0):
        self.q, self.s, self.s_th2, self.s_eps2 = q, s, s_th2, s_eps2
        self.V = s_th2 + s_eps2
        self.k = s_th2 / self.V
        self.rho = s * s / (s * s + self.V)
        self.tau = np.sqrt(s * s * self.V / (s * s + self.V))
        self.beta = beta_dye(q)
        self.d = np.linspace(-width * s, width * s, n)
        lg = norm.logpdf(self.d, 0, s)
        self.lg = lg - logsumexp(lg)
        self.G = np.exp(self.lg)

    # ---------- closed forms (Theorem 1, Corollary 2, Proposition 5) ----------
    def dstar(self, g):
        return self.beta * self.s * np.sqrt(1 + self.s ** 2 / self.V) + g * self.s ** 2 / self.s_th2

    def g0(self):
        return (self.beta * self.s_th2 / self.s) * (1 - np.sqrt(1 + self.s ** 2 / self.V))

    def zbar(self, g):
        """Silent hiring bar in z-units at the equilibrium: zbar = -d* - g V / s_th^2."""
        return -self.dstar(g) - g * self.V / self.s_th2

    # ---------- silent-pool objects ----------
    def logpool(self, t):
        return self.lg + np.log(np.where(self.d < t, 1.0, 1.0 - self.q))

    def post_d(self, z, lw):
        lp = lw - 0.5 * (z + self.d) ** 2 / self.V
        lp -= logsumexp(lp)
        return np.exp(lp)

    def Ed(self, z, lw):
        return (self.post_d(z, lw) * self.d).sum()

    def Vard(self, z, lw):
        p = self.post_d(z, lw)
        e = (p * self.d).sum()
        return (p * self.d ** 2).sum() - e * e

    def E_d_silent(self, t):
        lw = self.logpool(t)
        w = np.exp(lw - logsumexp(lw))
        return (w * self.d).sum()

    # ---------- equilibrium by brute force (checks the closed form) ----------
    def dstar_numerical(self, g):
        target = -g * self.V / self.s_th2

        def F(t):
            lw = self.logpool(t)
            zb = brentq(lambda z: z + self.Ed(z, lw) - target, target - 40, target + 40)
            return self.Ed(zb, lw) - t

        return brentq(F, -25 * self.s, 25 * self.s)

    # ---------- disclosure probabilities ----------
    def p_disclose_exante(self, g):
        return 1 - norm.cdf(self.dstar(g) / self.s)          # among informed agents

    def p_disclose_expost(self):
        # after seeing z the informed agent discloses iff d > t(z) = -rho z + beta tau   (independent of g)
        pd = 1 - norm.cdf(((self.beta * self.tau - self.d) / self.rho + self.d) / np.sqrt(self.V))
        return (self.G * pd).sum()

    # ---------- regimes: hiring probability and evaluator's residual uncertainty ----------
    def regime(self, g, which):
        V, k, q, s, sth2 = self.V, self.k, self.q, self.s, self.s_th2
        PD = norm.cdf(g * np.sqrt(V) / sth2)                   # hired prob. when context is known
        k0 = sth2 / (V + s * s)
        if which == "mandatory":
            hire = q * PD + (1 - q) * norm.cdf(g * np.sqrt(V + s * s) / sth2)
            resid = q * sth2 * (1 - k) + (1 - q) * sth2 * (1 - k0)
            return hire, resid
        if which == "banned":
            return norm.cdf(g * np.sqrt(V + s * s) / sth2), sth2 * (1 - k0)
        # voluntary (ex ante) disclosure, closed-form threshold
        t, zb = self.dstar(g), self.zbar(g)
        Psil = 1 - norm.cdf((zb + self.d) / np.sqrt(V))
        hire = q * (self.G * np.where(self.d >= t, PD, Psil)).sum() + (1 - q) * (self.G * Psil).sum()
        lw = self.logpool(t)
        w = np.exp(lw - logsumexp(lw))
        zs = np.linspace(-12 * max(s, 1), 12 * max(s, 1), 801)
        fz = np.array([(w * norm.pdf((z + self.d) / np.sqrt(V))).sum() for z in zs])
        fz /= fz.sum()
        ev = (fz * np.array([self.Vard(z, lw) for z in zs])).sum()
        p_disc = q * (1 - norm.cdf(t / s))
        resid = p_disc * sth2 * (1 - k) + (1 - p_disc) * (sth2 * (1 - k) + k * k * ev)
        return hire, resid

    # ---------- smooth payoffs  E[Phi((mu - c)/omega)] ----------
    def dstar_smooth(self, g, omega, nz=121):
        V, k = self.V, self.k
        zq = np.linspace(-6, 6, nz)
        wq = norm.pdf(zq)
        wq /= wq.sum()
        if np.isinf(omega):
            UD = g                                              # linear payoff: E[mu - c] = g under disclosure
        else:
            UD = (wq * norm.cdf((g + k * np.sqrt(V) * zq) / omega)).sum()

        def F(t):
            lw = self.logpool(t)
            zs = -t + np.sqrt(V) * zq
            mus = np.array([g + k * (z + self.Ed(z, lw)) for z in zs])
            US = (wq * mus).sum() if np.isinf(omega) else (wq * norm.cdf(mus / omega)).sum()
            return US - UD

        return brentq(F, -8 * self.s, 8 * self.s)


if __name__ == "__main__":
    M = Model()
    print("beta(q) =", round(M.beta, 4), " g0 =", round(M.g0(), 4))
    for g in [-1.0, 0.0, 1.0]:
        print(f"g={g:+.1f}: d* closed form {M.dstar(g):+.4f}, numerical {M.dstar_numerical(g):+.4f}")
    print("ex post disclosure prob (informed):", round(M.p_disclose_expost(), 4))
