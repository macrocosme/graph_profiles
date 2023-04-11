from math import sin
from astropy.units import c, km, degree


class Emission:
    self.delta_s = 0.2
    # w: pulse width
    # How to cancel a sine algebraicly?
    sin(self.W/4) = lambda alpha, beta: np.sqrt((sin(self.p/2)**2 - sin(beta/2)**2) / (sin(alpha) * sin(alpha + beta)))

    # p: angular radius, beta: angle between line of sight and magnetic angle, alpha: magnetic angle 
    self.p = lambda H, P: np.sqrt((9*math.pi*H)/(2*c*P))

    # Component width as a function of emission height
    self.w_p = lambda H, P: ((2.45 * degree) * self.delta_s) * np.sqrt((H * km)/(10*P))
    
    # Height of emission 
    self.H_v = lambda v, K, H_0: (K*v)**(-2/3) + H_0


    def __init__(self, p, alpha, beta):
        pass