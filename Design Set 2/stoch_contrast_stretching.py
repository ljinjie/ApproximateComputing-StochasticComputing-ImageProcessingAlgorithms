import numpy as np
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

n = 8 - trunc_bits
pow2n = int(2**n)

MAX_STATES = 32


def fsm_linear_gain(current_state, x, k):
    if current_state < MAX_STATES // 2:
        if x == 1:
            current_state = current_state + 1
        elif x == 0 and current_state != 0 and k == 1:
            current_state = current_state - 1
    else:
        if x == 0:
            current_state = current_state - 1
        elif x == 1 and current_state != MAX_STATES - 1 and k == 1:
            current_state = current_state + 1

    if current_state < MAX_STATES // 2:
        return current_state, 0
    else:
        return current_state, 1


def fsm_linear_gain_init(Is, Ks):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(8 * MAX_STATES):
        current_state, y = fsm_linear_gain(current_state, Is[init_idx], Ks[init_idx])
    return current_state


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


if __name__ == '__main__':

    infilename = 'contrast_stretching_original.jpg'
    outfilename = 'CS_S2_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc

    a = np.amin(image)
    b = np.amax(image)
    pc = (a + b) / (2 * 256 / trunc - 2)
    pk = ((2 * 256 / trunc - 2) + a - b) / ((2 * 256 / trunc - 2) - a + b)

    Rs = np.random.randint(0, 256 / trunc, (4, *image.shape, pow2n), dtype='uint8')
    Xs = (Rs[0] < image[..., np.newaxis])
    # Xs = Xs.astype(np.uint8)

    Cs = np.less(Rs[1], pc * 256 / trunc)
    Cs = Cs.astype(np.uint8)

    Ss = np.less(Rs[2], 128 / trunc)
    Ss = Ss.astype(np.uint8)

    Ks = np.less(Rs[3], pk * 256 / trunc)
    Ks = Ks.astype(np.uint8)

    Is = Ss * Xs + (1 - Ss) * Cs

    # Initialize output array
    Os = np.zeros(Xs.shape, dtype='uint8')

    for i in range(Is.shape[0]):
        for j in range(Is.shape[1]):
            current_state = fsm_linear_gain_init(Is[i, j], Ks[i, j])
            for r_idx in range(Is.shape[2]):
                current_state, Os[i, j, r_idx] = fsm_linear_gain(current_state, Is[i, j, r_idx], Ks[i, j, r_idx])

    save_image(np.mean(Os, axis=-1) * 256, outfilename)
