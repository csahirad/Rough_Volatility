import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class VolSurface(object):
    def __init__(self, strikes, maturities, implied_vols):
        self.strikes = strikes
        self.maturities = maturities
        self.implied_vols = implied_vols

    def plot_smile(self, T):
        """
        Plots a slice of the volatility surface given a time to maturity
        """
        vols = self.implied_vols[int(T * self.implied_vols.shape[0] / self.maturities[-1]), :]
        plt.plot(self.strikes, vols, 'r', lw=2)
        plt.xlabel(r'$k$', fontsize=16)
        plt.ylabel(r'$\sigma_{BS}(k,t=%.2f)$' % T, fontsize=16)
        plt.grid(True)
        plt.show()
