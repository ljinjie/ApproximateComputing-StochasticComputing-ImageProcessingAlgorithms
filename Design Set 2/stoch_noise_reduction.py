import numpy as np
from PIL import Image

trunc_bits = 4
trunc = int(2**trunc_bits)

# n = max(8 - trunc_bits, 4)
n = 8
pow2n = int(2**n)

MAX_STATES = 16

# Generate SNs for select lines
Rss = np.random.randint(0, 2, pow2n, dtype='uint8')
Ss = np.equal(Rss, 1)
Ss_neg = np.logical_not(Ss)
Ss = Ss.astype(np.uint8)
Ss_neg = Ss_neg.astype(np.uint8)


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


def fsm_tanh_init(Xs):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(MAX_STATES):
        current_state, y = fsm_tanh(current_state, Xs[init_idx])
    return current_state


def stoch_sorting_min(As, Bs):
    t_1 = Ss * As + Ss_neg * (1 - Bs)
    current_state = fsm_tanh_init(t_1)
    for t_idx in range(t_1.size):
        current_state, t_1[t_idx] = fsm_tanh(current_state, t_1[t_idx])

    min_of_AB = t_1 * Bs + (1 - t_1) * As
    return min_of_AB


def stoch_sorting_max(As, Bs):
    t_1 = Ss * As + Ss_neg * (1 - Bs)
    current_state = fsm_tanh_init(t_1)
    for t_idx in range(t_1.size):
        current_state, t_1[t_idx] = fsm_tanh(current_state, t_1[t_idx])

    max_of_AB = t_1 * As + (1 - t_1) * Bs
    return max_of_AB


def stoch_median_filter(Xs, i, j):
    a = stoch_sorting_min(stoch_sorting_min(Xs[i, j], Xs[i, j + 1]), stoch_sorting_min(Xs[i, j + 2], Xs[i + 1, j]))
    c = stoch_sorting_min(stoch_sorting_max(Xs[i, j], Xs[i, j + 1]), stoch_sorting_max(Xs[i, j + 2], Xs[i + 1, j]))
    d = stoch_sorting_max(stoch_sorting_max(Xs[i, j], Xs[i, j + 1]), stoch_sorting_max(Xs[i, j + 2], Xs[i + 1, j]))

    e = stoch_sorting_min(stoch_sorting_min(Xs[i + 1, j + 1], Xs[i + 1, j + 2]),
                          stoch_sorting_min(Xs[i + 2, j], Xs[i + 2, j + 1]))
    f = stoch_sorting_max(stoch_sorting_min(Xs[i + 1, j + 1], Xs[i + 1, j + 2]),
                          stoch_sorting_min(Xs[i + 2, j], Xs[i + 2, j + 1]))
    g = stoch_sorting_min(stoch_sorting_max(Xs[i + 1, j + 1], Xs[i + 1, j + 2]),
                          stoch_sorting_max(Xs[i + 2, j], Xs[i + 2, j + 1]))
    h = stoch_sorting_max(stoch_sorting_max(Xs[i + 1, j + 1], Xs[i + 1, j + 2]),
                          stoch_sorting_max(Xs[i + 2, j], Xs[i + 2, j + 1]))

    p = stoch_sorting_min(stoch_sorting_max(c, d), stoch_sorting_max(g, h))
    q = stoch_sorting_max(stoch_sorting_min(c, d), stoch_sorting_min(f, g))

    k = stoch_sorting_min(stoch_sorting_min(stoch_sorting_min(d, h), q), stoch_sorting_max(p, stoch_sorting_max(a, e)))
    l = stoch_sorting_max(stoch_sorting_min(stoch_sorting_min(d, h), q), stoch_sorting_max(p, stoch_sorting_max(a, e)))

    return stoch_sorting_min(l, stoch_sorting_max(Xs[i + 2, j + 2], k))


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


if __name__ == '__main__':

    infilename = 'noise_reduction_original.jpg'
    outfilename = 'NR_S2_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc

    Is = np.random.randint(0, pow2n // trunc, (*image.shape, pow2n), dtype='uint8')
    Is = (Is < image[..., np.newaxis])
    Is_neg = np.logical_not(Is)
    Is = Is.astype(np.uint8)
    Is_neg = Is_neg.astype(np.uint8)

    # Initialize output array
    Os = np.zeros((Is.shape[0] - 2, Is.shape[1] - 2, Is.shape[2]), dtype='uint8')

    for i in range(Os.shape[0]):
        print('Working on pixel row ' + str(i))
        for j in range(Os.shape[1]):
            # print('Working on pixel [' + str(i) + ', ' + str(j) + ']')
            Os[i, j] = stoch_median_filter(Is, i, j)

    save_image(np.sum(Os, axis=-1) * 256 / pow2n, outfilename)
