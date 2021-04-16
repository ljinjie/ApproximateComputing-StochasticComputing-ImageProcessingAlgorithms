import numpy as np
from PIL import Image

stoch_bits = 3
stoch = int(2**stoch_bits)
pow2n = stoch


MAX_STATES = 16

# Generate SNs for select lines
# Rss = np.random.randint(0, 2, pow2n, dtype='uint8')

Rss = np.random.permutation(pow2n)
Rss = (Rss < pow2n / 2)

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
        current_state, y = fsm_tanh(current_state, Xs[init_idx % Xs.size])
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


def median_sorting(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    a = min(min(a11, a12), min(a13, a21))
    b = max(min(a11, a12), min(a13, a21))
    c = min(max(a11, a12), max(a13, a21))
    d = max(max(a11, a12), max(a13, a21))

    e = min(min(a22, a23), min(a31, a32))
    f = max(min(a22, a23), min(a31, a32))
    g = min(max(a22, a23), max(a31, a32))
    h = max(max(a22, a23), max(a31, a32))

    i = min(max(c, d), max(g, h))
    j = max(min(c, d), min(f, g))

    k = min(min(min(d, h), j), max(i, max(a, e)))
    l = max(min(min(d, h), j), max(i, max(a, e)))

    return min(l, max(a33, k))


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    infilename = 'noise_reduction_original.jpg'
    outfilename = 'perm_NR_S3_' + str(stoch_bits) + '_stochastic_bits.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image_conv = (image // stoch) * stoch
    image_stoch = image % stoch

    # conventional part
    output_image_conv = np.zeros((image.shape[0] - 2, image.shape[1] - 2))

    for m in range(image.shape[0] - 2):
        for n in range(image.shape[1] - 2):
            output_image_conv[m, n] = median_sorting(image_conv[m, n], image_conv[m, n + 1], image_conv[m, n + 2],
                                                image_conv[m + 1, n], image_conv[m + 1, n + 1], image_conv[m + 1, n + 2],
                                                image_conv[m + 2, n], image_conv[m + 2, n + 1], image_conv[m + 2, n + 2])

    # stochastic part
    # Is = np.random.randint(0, pow2n, (*image_stoch.shape, pow2n), dtype='uint8')
    Is = np.zeros((*image_stoch.shape, pow2n), dtype='uint8')
    for i in range(Is.shape[0]):
        for j in range(Is.shape[1]):
            Is[i, j] = np.random.permutation(pow2n)

    Is = (Is < image_stoch[..., np.newaxis])
    Is_neg = np.logical_not(Is)
    Is = Is.astype(np.uint8)
    Is_neg = Is_neg.astype(np.uint8)

    # Initialize output array
    output_image_stoch = np.zeros((Is.shape[0] - 2, Is.shape[1] - 2, Is.shape[2]), dtype='uint8')

    for i in range(output_image_stoch.shape[0]):
        for j in range(output_image_stoch.shape[1]):
            # print('Working on pixel [' + str(i) + ', ' + str(j) + ']')
            output_image_stoch[i, j] = stoch_median_filter(Is, i, j)

    output_image_stoch = np.mean(output_image_stoch, axis=-1) * (stoch - 1)
    save_image(output_image_conv + output_image_stoch, outfilename)
