# import some important libraries
import numpy as np
import matplotlib.pyplot as plt

MAX_STATES = 16


def fsm_absolute_value(current_state, x):
    if x == 1 and current_state != MAX_STATES - 1:
        if current_state < MAX_STATES / 2 and current_state % 2 == 0:
            return current_state + 1, 1
        elif current_state < MAX_STATES / 2 and current_state % 2 == 1:
            return current_state + 1, 0
        elif current_state >= MAX_STATES / 2 and current_state % 2 == 0:
            return current_state + 1, 0
        elif current_state >= MAX_STATES / 2 and current_state % 2 == 1:
            return current_state + 1, 1
    elif x == 1 and current_state == MAX_STATES - 1:
        return current_state, 1
    elif x == 0 and current_state != 0:
        if current_state < MAX_STATES / 2 and current_state % 2 == 0:
            return current_state - 1, 1
        elif current_state < MAX_STATES / 2 and current_state % 2 == 1:
            return current_state - 1, 0
        elif current_state >= MAX_STATES / 2 and current_state % 2 == 0:
            return current_state - 1, 0
        elif current_state >= MAX_STATES / 2 and current_state % 2 == 1:
            return current_state - 1, 1
    else:
        return current_state, 1


def fsm_absolute_value_init(runs, i, Xs):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(MAX_STATES):
        current_state, y = fsm_absolute_value(current_state, Xs[runs, i, init_idx])
    return current_state


if __name__ == '__main__':
    # Replacing large for loops with built-in or library functions is a good way to speed up code.
    # We will make the same plot as last time
    n = 8
    runs = 1
    pow2n = int(2**n)
    N = pow2n  # SN length

    pxs = np.arange(pow2n + 1) / pow2n
    pxs = 2 * pxs - 1

    Rs = np.random.randint(-pow2n, pow2n, (runs, *pxs.shape, pow2n))
    Bxs = pxs * pow2n
    Xs = Rs < Bxs[:, np.newaxis]
    Xs = Xs.astype(int)

    np.set_printoptions(threshold=np.inf)

    results = np.zeros(Rs.shape)

    current_state = fsm_absolute_value_init(0, 0, Xs)

    for r_idx in range(runs):
        for i in range(Xs.shape[1]):
            for j in range(Xs.shape[2]):
                # current_state = fsm_absolute_value_init(r_idx, i, Xs)
                current_state, results[r_idx, i, j] = fsm_absolute_value(current_state, Xs[r_idx, i, j])

    FONTSIZE = 14
    plt.plot(pxs, 2 * np.mean(np.mean(results, axis=-1), axis=0) - 1, 'bo')
    plt.xlabel("X", fontsize=FONTSIZE)
    plt.ylabel("Y", fontsize=FONTSIZE)
    plt.title('Absolute Value')
    plt.tight_layout()
    plt.show()
