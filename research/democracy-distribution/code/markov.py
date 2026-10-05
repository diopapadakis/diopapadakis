import numpy as np
from model import Econ

class Markov:
    """Markov equilibrium of the Benabou economy with regime transitions.
    r=0: autocracy (tau_A), r=1: democracy (tau_D).
    Coups: in D, w.p. p the elite (rank z) may coup at private cost psi.
    Revolutions: in A, w.p. q the median (x=0) may revolt at cost mu; elite may pre-empt by a
      one-period concession (tau_D, stay A) if allowed, else democratize.
    Voluntary: in A every period the elite may democratize at private cost psiV (None = not allowed).
    Coup option for the elite every period in D can be emulated with p=1."""
    def __init__(s, e, p=0.3, psi=0.5, q=0.3, mu=0.35, conc=True, psiV=None, grid=(0.02, 3.5, 700)):
        s.e, s.p, s.psi, s.q, s.mu, s.conc, s.psiV = e, p, psi, q, mu, conc, psiV
        s.G = np.linspace(*grid)
        s.W = np.zeros((2, s.G.size)); s.Bv = np.zeros((2, s.G.size))
        for r, t in enumerate([e.tauA, e.tauD]):   # start from permanent-regime values
            s.W[r] = e.om*e.d(t)/(1-e.dl) + e.Pi(t)*(s.G + e.dl*e.s2/(1-e.dl))
            s.Bv[r] = e.B(t)
    def post(s, D2, t, r):
        e = s.e; rho = e.rho(t); D2n = rho**2*D2 + e.s2
        W = e.om*(e.d(t) + e.g1(t)*D2) + e.dl*np.interp(D2n, s.G, s.W[r])
        B = (1-t) + e.dl*rho*np.interp(D2n, s.G, s.Bv[r])
        return W, B
    def outcomes(s, D2):
        """returns list over r_prev of list of (prob, W, B, label) for each D2 (vectorized)"""
        e = s.e; zD = e.z*np.sqrt(D2)
        WA, BA = s.post(D2, e.tauA, 0)      # autocracy, normal policy
        WD, BD = s.post(D2, e.tauD, 1)      # democracy
        WC, BC = s.post(D2, e.tauD, 0)      # autocratic concession
        EA, ED, EC = WA + BA*zD, WD + BD*zD, WC + BC*zD   # elite values
        # --- democracy, coup opportunity
        coup = (EA - s.psi) > ED
        # --- autocracy: elite's best non-threat action
        vol = np.zeros_like(D2, bool) if s.psiV is None else (ED - s.psiV) > EA
        # --- autocracy under revolution threat (masses = median, x=0)
        revV = WD - s.mu
        ok_none = WA >= revV
        ok_conc = (WC >= revV) & s.conc
        # elite picks: if voluntary democratization preferred, do it; else none if ok, else concession if ok & EC>=ED-psiV.., else democratize
        thr_choice = np.where(vol, 2, np.where(ok_none, 0, np.where(ok_conc & (EC >= ED), 1, 2)))
        return dict(WA=WA, BA=BA, WD=WD, BD=BD, WC=WC, BC=BC, coup=coup, vol=vol, thr=thr_choice)
    def solve(s, tol=1e-9, maxit=5000, damp=0.5, verbose=False):
        for it in range(maxit):
            o = s.outcomes(s.G)
            # D_prev
            WDn = s.p*np.where(o['coup'], o['WA'], o['WD']) + (1-s.p)*o['WD']
            BDn = s.p*np.where(o['coup'], o['BA'], o['BD']) + (1-s.p)*o['BD']
            # A_prev, no threat
            Wn0 = np.where(o['vol'], o['WD'], o['WA']); Bn0 = np.where(o['vol'], o['BD'], o['BA'])
            Wthr = np.choose(o['thr'], [o['WA'], o['WC'], o['WD']]); Bthr = np.choose(o['thr'], [o['BA'], o['BC'], o['BD']])
            WAn = s.q*Wthr + (1-s.q)*Wn0; BAn = s.q*Bthr + (1-s.q)*Bn0
            newW = np.vstack([WAn, WDn]); newB = np.vstack([BAn, BDn])
            err = max(np.abs(newW-s.W).max(), np.abs(newB-s.Bv).max())
            s.W = damp*s.W + (1-damp)*newW; s.Bv = damp*s.Bv + (1-damp)*newB
            if err < tol: 
                if verbose: print("converged", it, err)
                return True
        print("NOT converged, err", err); return False
    def thresholds(s):
        o = s.outcomes(s.G); G = np.sqrt(s.G)
        def edges(mask):
            idx = np.where(np.diff(mask.astype(int)) != 0)[0]
            return [round(float((G[i]+G[i+1])/2), 3) for i in idx]
        return dict(coup=edges(o['coup']), vol=edges(o['vol']), thr_none=edges(o['thr'] == 0), thr_dem=edges(o['thr'] == 2))
    def simulate(s, D2_0, r0, T, rng):
        e = s.e; D2, r = D2_0, r0
        out = []
        for t in range(T):
            o = s.outcomes(np.array([D2]))
            ev = ''
            if r == 1:
                if rng.random() < s.p and o['coup'][0]: r, ev = 0, 'coup'
            else:
                if rng.random() < s.q:
                    c = o['thr'][0]
                    if c == 2: r, ev = 1, 'democratization(threat)'
                    elif c == 1: ev = 'concession'
                elif o['vol'][0]: r, ev = 1, 'democratization(voluntary)'
            t_pol = e.tauD if (r == 1 or ev == 'concession') else e.tauA
            out.append((t, r, np.sqrt(D2), ev))
            D2 = e.rho(t_pol)**2*D2 + e.s2
        return out
