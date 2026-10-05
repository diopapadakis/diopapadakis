import numpy as np

class Econ:
    def __init__(s, al=0.2, be=0.45, dl=0.5, tauA=0.0, tauD=0.25, theta=1.0, s2=0.47, z=2.326):
        s.al, s.be, s.dl, s.tauA, s.tauD, s.theta, s.s2, s.z = al, be, dl, tauA, tauD, theta, s2, z
        s.a = al+be
        s.om = (1-dl*al)/(1-dl*s.a)
    def d(s, t):   return -s.theta*t**2/2
    def g1(s, t):  return t*(2-t)/2
    def rho(s, t): return s.al + s.be*(1-t)
    def Pi(s, t):  return s.om*s.g1(t)/(1-s.dl*s.rho(t)**2)
    def B(s, t):   return (1-t)/(1-s.dl*s.rho(t))
    def Dstar2(s, t): return s.s2/(1-s.rho(t)**2)
    # permanent-regime surpluses (D minus A)
    def SM(s, D2):
        return s.om*(s.d(s.tauD)-s.d(s.tauA))/(1-s.dl) + (s.Pi(s.tauD)-s.Pi(s.tauA))*(D2 + s.dl*s.s2/(1-s.dl))
    def SE(s, Dl):
        return s.SM(Dl**2) - (s.B(s.tauA)-s.B(s.tauD))*s.z*Dl
    def Dhat(s):
        return (s.B(s.tauA)-s.B(s.tauD))*s.z/(2*(s.Pi(s.tauD)-s.Pi(s.tauA)))

if __name__ == "__main__":
    e = Econ()
    print("rhoA %.4f rhoD %.4f omega %.4f" % (e.rho(e.tauA), e.rho(e.tauD), e.om))
    print("Pi_D %.4f Pi_A %.4f B_A %.4f B_D %.4f" % (e.Pi(e.tauD), e.Pi(e.tauA), e.B(e.tauA), e.B(e.tauD)))
    print("Delta*_A %.3f Delta*_D %.3f" % (np.sqrt(e.Dstar2(e.tauA)), np.sqrt(e.Dstar2(e.tauD))))
    for z in [2.326, 1.645, 1.282, 0.674]:
        e.z = z
        print(f"z={z}: Dhat={e.Dhat():.3f}  SE(D*_D)={e.SE(np.sqrt(e.Dstar2(e.tauD))):.3f}  SE(D*_A)={e.SE(np.sqrt(e.Dstar2(e.tauA))):.3f}  SM(D*_A^2)={e.SM(e.Dstar2(e.tauA)):.3f}")
