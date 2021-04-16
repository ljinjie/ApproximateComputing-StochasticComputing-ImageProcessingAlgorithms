import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MAX_STATES = 8


def fsm_tanh(current_state, x):
    if x == 1 and current_state != MAX_STATES - 1:
        if current_state < MAX_STATES // 2:
            return current_state + 1, 0
        else:
            return current_state + 1, 1
    elif x == 1 and current_state == MAX_STATES - 1:
        return current_state, 1
    elif x == 0 and current_state != 0:
        if current_state < MAX_STATES // 2:
            return current_state - 1, 0
        else:
            return current_state - 1, 1
    elif x == 0 and current_state == 0:
        return current_state, 0


def fsm_tanh_init(runs, ts):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(runs):
        current_state, y = fsm_tanh(current_state, ts[init_idx % ts.size])
    return current_state


def maj(ts):
    outputs = np.zeros(16)
    for i in range(16):
        sum = 0
        for j in range(16):
            sum = sum + ts[i + j * 16]
        outputs[i] = 1 if sum > 7 else 0
    return np.sum(outputs)


if __name__ == '__main__':
    ss_dis = pd.read_excel('8_state_fsm_steady_state_distribution.xlsx', header=None)
    ss_dis = ss_dis.to_numpy()

    n = 4
    runs = 100
    pow2n = int(2**n)

    xys = np.random.randint(0, pow2n, (2, runs))
    theo_results = np.greater(xys[0], xys[1])
    Ks = np.random.randint(0, pow2n, (3, runs, 256))
    Xs = Ks[0] < xys[0, :, np.newaxis]
    Ys = Ks[1] < xys[1, :, np.newaxis]
    Ss = np.less(Ks[2], pow2n / 2)

    Ts = Ss * Xs + (1 - Ss) * (1 - Ys)

    results = np.zeros(Ts.shape)

    for r_idx in range(runs):
        est_val = int(maj(Ts[r_idx]))
        fsm_choice = ss_dis[est_val]
        set_number = 0
        for ss_idx in range(MAX_STATES):
            for f_idx in range(fsm_choice[ss_idx]):
                print('Using set #' + str(set_number + 1))
                current_state = ss_idx
                for i in range(16):
                    current_state, results[r_idx, set_number * 16 + i] = fsm_tanh(current_state,
                                                                                  Ts[r_idx, set_number * 16 + i])
                set_number = set_number + 1

    results_mean = np.mean(results, axis=-1)
    mses = np.square(results_mean - theo_results)

    x = np.arange(runs)

    FONTSIZE = 14
    plt.plot(x, mses, 'bo')
    plt.xlabel("Warm-up Cycles", fontsize=FONTSIZE)
    plt.ylabel("MSE", fontsize=FONTSIZE)
    plt.title('MSEs in Comparator for Different Warm-up Cycles')
    plt.tight_layout()
    plt.show()
