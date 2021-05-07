import math
from scipy.stats import norm


def BS_vega(S, K, T, r, sigma_bs):
    """ Compute the Black Sholes vega of a vanilla option """
    d1 = 1 / (sigma_bs * math.sqrt(T)) * (math.log(S / K) + (r + (sigma_bs ** 2) / 2) * T)
    return S * norm.pdf(d1) * math.sqrt(T)
