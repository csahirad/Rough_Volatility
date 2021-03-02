from utils import *
from scipy.interpolate import interp1d


class RBergomi(object):
    """
    Class for generating paths of the rBergomi model.
    """

    def __init__(self, xis, n=100, N=1000, S0=1.0, T=1.0, a=-0.4, eta=1.0, rho=-0.9):
        """
        Constructor for class.
        """
        # Basic assignments
        self.S0 = S0  # Initial stock price
        self.T = T  # Maturity in years
        self.n = n  # Granularity (steps per year)
        self.dt = 1.0 / self.n  # Step size
        self.s = int(self.n * self.T)  # Steps
        self.t = np.linspace(0, self.T, 1 + self.s).T  # Time grid
        self.a = a  # Alpha
        self.N = N  # Paths

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n)

        # Parameters for variance process
        self.xis = xis  # Nodes of the initial forward variance curve
        self.eta = eta
        self.rho = rho

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.N, 1 + self.s))  # Exact integrals

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:, i] = dW[:, i - 1, 1]  # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s)  # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a) / self.n, self.a)

        X = dW[:, :, 0]  # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0, :]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.N):
            GX[i, :] = np.convolve(G, X[i, :])

        # Extract appropriate part of convolution
        Y2 = GX[:, :1 + self.s]  # Riemann sums

        # Finally contruct and return full process
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2):
        """
        Constructs correlated price Brownian increments, dB.
        """
        dB = self.rho * dW1[:, :, 0] + np.sqrt(1 - self.rho ** 2) * dW2
        return dB

    def XI(self, interp_method='pwc'):
        """
        Constructs initial forward variance curve using a specified interpolation scheme
        """
        tenors = [x[0] for x in self.xis]
        forward_variances = [x[1] for x in self.xis]
        if interp_method == 'pwc':
            XI = interp1d(tenors, forward_variances, 'nearest')
        elif interp_method == 'log-linear':
            log_forward_variances = np.log(forward_variances)
            lin_interp = interp1d(tenors, log_forward_variances, 'linear')
            XI = lambda zz: np.exp(lin_interp(zz))
        elif interp_method == 'log-cubic-spline':
            log_forward_variances = np.log(forward_variances)
            cubic_interp = interp1d(tenors, log_forward_variances, 'cubic')
            XI = lambda zz: np.exp(cubic_interp(zz))
        return XI

    def V(self, Y, XI):
        """
        rBergomi variance process.
        """
        a = self.a
        t = self.t
        V = XI(t) * np.exp(self.eta * Y - 0.5 * self.eta ** 2 * t ** (2 * a + 1))
        return V

    def S(self, V, dB):
        """
        rBergomi price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = self.S0
        S[:, 1:] = self.S0 * np.exp(integral)
        return S
