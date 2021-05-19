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


def barrier(S, K, B, o='call', Type='up-and-out'):
    """
    Returns an array of prices of a barrier option given an array of stock price paths, an array of stikes, and a barrier level
    """
    S_max = np.max(S, axis=1)
    S_min = np.min(S, axis=1)
    ST = S[:, -1][:, np.newaxis]
    max_below = (S_max < B)[:, np.newaxis]
    max_above = (S_max > B)[:, np.newaxis]
    min_above = (S_min > B)[:, np.newaxis]
    min_below = (S_min < B)[:, np.newaxis]

    w = 1.
    if o == 'put':
        w = -1.

    # Payoff of an up-and-out barrier option
    payoffs = np.maximum(max_below * w * (ST - K), 0)

    if Type == 'up-and-in':
        payoffs = np.maximum(max_above * w * (ST - K), 0)
    elif Type == 'down-and-out':
        payoffs = np.maximum(min_above * w * (ST - K), 0)
    elif Type == 'down-and-in':
        payoffs = np.maximum(min_below * w * (ST - K), 0)
    return np.mean(payoffs, axis=0)[:, np.newaxis]


def cliquet(S, cap=5, freq=12, T=1):
    """
    Returns the price of a cliquet option where the payoff is the greater of zero, and the sum of returns at a specific
    frequency(e.g. monthly, quarterly, etc.), capped at a specified rate
    """

    return_sum = 0
    ts = np.linspace(0, S.shape[1] - 1, freq * T)
    for i in range(len(ts) - 1):
        return_sum += np.minimum(cap, S[:, int(ts[i+1])] / S[:, int(ts[i])]*100 - 100)
    payoffs = np.maximum(0, return_sum)
    return np.mean(payoffs, axis=0)


def autocallable(S, coupon=5, barrier_lvl=0.70, autocall_freq=4, T=1):
    """
    Returns the price of autocallable note
    """
    # Notes are autocallable if the closing level of the asset on an observation date is at or above its inital level.
    # If called, the investor will receive a coupon. If the notes are not called, the notes provide principal protection
    # at maturity if the asset return is greater than the barrier level. If the asset return is equal to, or less than,
    # the barrier, the investor will be fully exposed to any negative performance.

    observation_ts = np.linspace(0, S.shape[1] - 1, autocall_freq * T + 1)
    payoffs = []
    # Iterate over each path
    for i in range(S.shape[0]):
        S0 = S[i, 0]
        ST = S[i, -1]
        # Iterate over each observation date
        for j in range(1, len(observation_ts)):
            if S[i, int(observation_ts[j])] > S0:
                payoffs.append(100 + coupon * j)
                break
            elif j < len(observation_ts) - 1:
                continue
            elif ST > barrier_lvl * S0:
                payoffs.append(100)
            else:
                payoffs.append(ST/S0*100)
    return np.mean(payoffs)
