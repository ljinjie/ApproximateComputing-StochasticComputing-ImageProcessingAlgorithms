# import some important libraries
import numpy as np
import matplotlib.pyplot as plt

MAX_STATES = 64


def fsm_linear_gain(current_state, x, k):
    if current_state < MAX_STATES // 2:
        if x == 1:
            return current_state + 1, 0
        elif x == 0 and current_state == 0:
            return current_state, 0
        elif x == 0 and current_state != 0 and k == 0:
            return current_state, 0
        elif x == 0 and current_state != 0 and k == 1:
            return current_state - 1, 0
    else:
        if x == 0:
            return current_state - 1, 1
        elif x == 1 and current_state == MAX_STATES - 1:
            return current_state, 1
        elif x == 1 and current_state != MAX_STATES - 1 and k == 0:
            return current_state, 1
        elif x == 1 and current_state != MAX_STATES - 1 and k == 1:
            return current_state + 1, 1


def fsm_linear_gain_init(cycles, i, Is, Ks):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(cycles + 1):
        current_state, y = fsm_linear_gain(current_state, Is[cycles, i, init_idx], Ks[cycles, i, init_idx])
    return current_state


if __name__ == '__main__':
    # Replacing large for loops with built-in or library functions is a good way to speed up code.
    # We will make the same plot as last time
    n = 8
    runs = 4 * MAX_STATES
    pc = 0.5
    pk = 0.75
    pow2n = int(2**n)
    N = pow2n  # SN length

    pxs = np.arange(pow2n + 1) / pow2n
    pxs = 2 * pxs - 1

    Rs = np.random.randint(-pow2n, pow2n, (4, runs, *pxs.shape, pow2n))
    Bxs = pxs * pow2n
    Xs = Rs[0] < Bxs[:, np.newaxis]
    pcs = np.less(Rs[1], (2 * pc - 1) * pow2n)
    Cs = pcs.astype(int)

    pss = np.less(Rs[2], 0)
    pss_neg = np.logical_not(pss)
    Ss = pss.astype(int)
    Ss_neg = pss_neg.astype(int)

    Is = Ss * Xs + Ss_neg * Cs

    pks = np.less(Rs[3], (2 * pk - 1) * pow2n)
    Ks = pks.astype(int)

    results = np.zeros(Is.shape)
    mses = np.zeros(runs)
    theo_results = np.arange(pow2n + 1) / pow2n

    for t_idx in range(theo_results.size):
        if theo_results[t_idx] < max(0.0, pc - (1 - pk)/(1 + pk)):
            theo_results[t_idx] = 0
        elif theo_results[t_idx] >= max(0.0, pc - (1 - pk)/(1 + pk)) and theo_results[t_idx] <= min(1.0, pc + (1 - pk)/(1 + pk)):
            theo_results[t_idx] = 0.5 * (1 + pk) * (theo_results[t_idx] - pc) / (1 - pk) + 0.5
        else:
            theo_results[t_idx] = 1

    for r_idx in range(runs):
        print('Running with ' + str(r_idx + 1) + ' warm-up cycles')

        for i in range(Is.shape[1]):
            for j in range(Is.shape[2]):
                current_state = fsm_linear_gain_init(r_idx, i, Is, Ks)
                current_state, results[r_idx, i, j] = fsm_linear_gain(current_state, Is[r_idx, i, j], Ks[r_idx, i, j])
        mses[r_idx] = np.mean(np.square(np.mean(results[r_idx], axis=-1) - theo_results))

    x = np.arange(1, runs + 1)

    FONTSIZE = 14
    # plt.plot(pxs, np.mean(np.mean(results, axis=-1), axis=0), 'bo')
    plt.plot(x, mses, 'bo')
    plt.xlabel("Warm-up Cycles", fontsize=FONTSIZE)
    plt.ylabel("MSE", fontsize=FONTSIZE)
    plt.title('MSEs in Linear Gain for Pc = ' + str(pc) + ', Pk = ' + str(pk))
    plt.tight_layout()
    plt.show()
