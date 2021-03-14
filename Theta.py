import random
import numpy as np


class Theta(object):
    """ Class for vector of model parameters"""

    def __init__(self, xi_ub, xi_lb, eta_ub, eta_lb, rho_ub, rho_lb, alpha_ub, alpha_lb):
        """Randomly assigns the model parameters to a value within the specified bounds"""
        self.xi0 = random.uniform(xi_lb, xi_ub)
        self.xi1 = random.uniform(xi_lb, xi_ub)
        self.xi2 = random.uniform(xi_lb, xi_ub)
        self.xi3 = random.uniform(xi_lb, xi_ub)
        self.xi4 = random.uniform(xi_lb, xi_ub)
        self.xi5 = random.uniform(xi_lb, xi_ub)
        self.xi6 = random.uniform(xi_lb, xi_ub)
        self.xi7 = random.uniform(xi_lb, xi_ub)
        self.xi8 = random.uniform(xi_lb, xi_ub)
        self.eta = random.uniform(eta_lb, eta_ub)
        self.rho = random.uniform(rho_lb, rho_ub)
        self.alpha = random.uniform(alpha_lb, alpha_ub)

    def to_array(self):
        return np.array([self.xi0, self.xi1, self.xi2, self.xi3, self.xi4, self.xi5, self.xi6, self.xi7, self.xi8,
                         self.eta, self.rho, self.alpha])
