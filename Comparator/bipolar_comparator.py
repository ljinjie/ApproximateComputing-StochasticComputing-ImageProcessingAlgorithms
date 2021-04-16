# import some important libraries
import numpy as np
import matplotlib.pyplot as plt

MAX_STATES = 32

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

def fsm_tanh_init():
    current_state = np.random.randint(0, MAX_STATES)
    rand_x = np.random.randint(0, 2, MAX_STATES // 2)
    for init_idx in range(len(rand_x)):
        current_state, y = fsm_tanh(current_state, rand_x[init_idx])
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

    # pxs and pys are 2D arrays where all possible pairings of elements from pxs and elements from pxs appear
    # uncomment the print statements to get a clearer idea of what these two arrays are
    pxs, pys = np.meshgrid(pxs, pxs)
    # print(pxs)
    # print(pys)

    # SNGs take in integers that relate to unipolar value in the following way: p = B / pow2n
    # Bxs = (pxs + 1) * pow2n / 2
    # Bys = (pys + 1) * pow2n / 2
    Bxs = pxs * pow2n
    Bys = pys * pow2n

    # [:,:,np.newaxis] takes Bxs from shape (17, 17) and makes its shape (17, 17, 1). This necessary or comparison with
    # R later will not work
    Bxs = Bxs[:, :, np.newaxis]

    # "..." can be used to replace one set of consecutive ":"'s when indexing an array. so the following statement is
    # equivalent to the last
    Bys = Bys[..., np.newaxis]

    # Some other useful variables. Notice how we calculate as much as possible outside of the for loop. and notice
    # how there is 2 fewer for loops compared to last simulation.
    R_shape = (3, *Bxs.shape, N)
    current_state = fsm_tanh_init()
    print(current_state)
    Zs = np.zeros((len(pxs), len(pys), pow2n))
    results = np.zeros((len(pxs), len(pys), runs))
    for r_idx in range(runs):
        print('Run #' + str(r_idx))

        # generates an array with shape "R_shape" and where each element is a random int in [0, pow2n)
        Rs = np.random.randint(-pow2n, pow2n, R_shape)
        # Rs = np.random.randint(0, pow2n, R_shape)

        Xs = Rs[0] < Bxs[..., np.newaxis]  # generate many X SNs.
        Ys = Rs[1] < Bys[..., np.newaxis]  # generate many Y SNs
        Ss = np.less(Rs[2], 0)             # generate select line SNs
        Ss_neg = np.logical_not(Ss)

        Ss = Ss.astype(int)
        Ss_neg = Ss_neg.astype(int)

        Ys = np.logical_not(Ys)
        Temps = Ss * Xs + Ss_neg * Ys

        for i in range(Temps.shape[0]):
            for j in range(Temps.shape[1]):
                for sn_idx in range(Temps.shape[3]):
                    current_state, Zs[i, j, sn_idx] = fsm_tanh(current_state, Temps[i, j, 0, sn_idx])

        results[..., :, r_idx] = np.mean(Zs, axis=-1)

    FONTSIZE = 14
    plt.pcolormesh(pxs, pys, np.mean(results, axis=-1), cmap="plasma")
    plt.colorbar().set_label("Comparison Results", rotation=270, labelpad=15, fontsize=FONTSIZE)
    plt.xlabel("$X$", fontsize=FONTSIZE)
    plt.ylabel("$Y$", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()
