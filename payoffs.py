import numpy as np


def vanilla(S, K, o='call'):
    """
    Returns an array of prices of a vanilla option given an array of stock price paths and an array of strikes
    """
    w = 1.
    if o == 'put':
        w = -1.
    ST = S[:, -1][:, np.newaxis]
    payoffs = np.maximum(w * (ST - K), 0)
    return np.mean(payoffs, axis=0)[:, np.newaxis]


def asian(S, K, o='call'):
    """
    Returns an array of prices of an asian option given an array of stock price paths and an array of strikes
    """
    w = 1.
    if o == 'put':
        w = -1.
    S_avg = np.mean(S, axis=1)[:, np.newaxis]
    payoffs = np.maximum(w * (S_avg - K), 0)
    return np.mean(payoffs, axis=0)[:, np.newaxis]


def barrier(S, K, B, o='up-and-out'):
    """
    Returns an array of prices of a barrier option given an array of stock price paths, an array of stikes, and a barrier level
    """
    S_max = np.max(S, axis=1)
    S_min = np.min(S, axis=1)
    ST = S[:, -1][:, np.newaxis]

    # Payoff of an up-and-out barrier option
    payoffs = np.maximum((S_max < B) * (ST - K), 0)

    if o == 'up-and-in':
        payoffs = np.maximum((S_max > B) * (ST - K), 0)
    elif o == 'down-and-out':
        payoffs = np.maximum((S_min > B) * (K - ST), 0)
    elif o == 'down-and-in':
        payoffs = np.maximum((S_min < B) * (K - ST), 0)
    return np.mean(payoffs, axis=0)[:, np.newaxis]


def cliquet(S, cap=0.05, freq=12, o='call'):
    """
    Returns the price of a cliquet option where the payoff is the greater of zero, and the sum of returns at a specific
    frequency(e.g. monthly, quarterly, etc.), capped at a specified rate
    """
    w = 1.
    if o == 'put':
        w = -1.
    return_sum = 0
    ts = np.linspace(0, S.shape[1] - 1, freq)
    for i in range(freq - 1):
        return_sum += np.minimum(cap, S[:, int(ts[i+1])] / S[:, int(ts[i])] - 1)
    payoffs = np.maximum(0, return_sum)
    return np.mean(payoffs, axis=0)


def autocallable():
    pass
