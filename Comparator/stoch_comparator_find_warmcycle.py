import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xw

MAX_STATES = 16


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


if __name__ == '__main__':
    n = 8
    runs = 64
    each_run = 7000
    pow2n = int(2**n)

    workbook = xw.Workbook('State Visit Counts 16 State.xlsx')
    worksheet = workbook.add_worksheet()

    xys = np.random.randint(0, pow2n, (2, runs, each_run), dtype='uint8')
    theo_results = np.greater(xys[0], xys[1])
    Ks = np.random.randint(0, pow2n, (3, runs, each_run, pow2n), dtype='uint8')
    Xs = Ks[0] < xys[0, ..., np.newaxis]
    Ys = Ks[1] < xys[1, ..., np.newaxis]
    Ss = np.less(Ks[2], pow2n / 2)
    Ss_neg = np.logical_not(Ss)
    Ys_neg = np.logical_not(Ys)

    Ts = Ss * Xs + Ss_neg * Ys_neg

    results = np.zeros(Ts.shape, dtype='uint8')

    for r_idx in range(runs):
        state_visits = np.zeros(MAX_STATES, dtype="uint16")
        for e_idx in range(each_run):
            current_state = fsm_tanh_init(r_idx * 16, Ts[r_idx, e_idx])
            state_visits[current_state] = state_visits[current_state] + 1
            for idx in range(pow2n):
                current_state, results[r_idx, e_idx, idx] = fsm_tanh(current_state, Ts[r_idx, e_idx, idx])
                state_visits[current_state] = state_visits[current_state] + 1

        print(state_visits)
        worksheet.write_row(r_idx, 0, state_visits)

    workbook.close()
    results_mean = np.mean(results, axis=-1)
    mses = np.mean(np.square(np.mean(results, axis=-1) - theo_results), axis=-1)

    x = np.arange(1, runs + 1)
    x = x * 16

    FONTSIZE = 14
    plt.plot(x, mses, 'bo')
    plt.xlabel("Warm-up Cycles", fontsize=FONTSIZE)
    plt.ylabel("MSE", fontsize=FONTSIZE)
    plt.title('MSEs in Comparator for Different Warm-up Cycles')
    plt.tight_layout()
    plt.show()
