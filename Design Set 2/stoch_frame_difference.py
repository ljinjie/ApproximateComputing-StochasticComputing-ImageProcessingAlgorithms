import numpy as np
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

n = 10 - trunc_bits
pow2n = int(2**n)

MAX_STATES = 16

# Threshold parameter
th = 80
pth = 0.5 + th / 510


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


def fsm_absolute_value_init(Xs):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(MAX_STATES):
        current_state, y = fsm_absolute_value(current_state, Xs[init_idx])
    return current_state


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


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


if __name__ == '__main__':

    infilename_1 = 'frame_difference_original_1.jpg'
    infilename_2 = 'frame_difference_original_2.jpg'
    outfilename = 'FD_S2_' + str(trunc_bits) + '_bits_truncated.jpg'
    image_1 = load_image(infilename_1)
    image_1 = image_1[..., 0]
    image_1 = image_1 // trunc

    image_2 = load_image(infilename_2)
    image_2 = image_2[..., 0]
    image_2 = image_2 // trunc

    Is = np.random.randint(0, 256 / trunc, (2, *image_1.shape, pow2n), dtype='uint8')
    Is_1 = (Is[0] < image_1[..., np.newaxis])
    Is_1 = Is_1.astype(np.uint8)
    Is_2 = (Is[1] < image_2[..., np.newaxis])
    Is_2 = Is_2.astype(np.uint8)

    # Initialize output array
    Os = np.zeros((Is[0].shape[0], Is[0].shape[1], Is[0].shape[2]), dtype='uint8')

    # Generate SNs for select lines
    Ss = np.random.randint(0, 2, (2, *image_1.shape, pow2n), dtype='uint8')
    Ss = np.equal(Ss, 1)
    Ss_1 = Ss[0].astype(np.uint8)
    Ss_2 = Ss[1].astype(np.uint8)

    Xs = Ss_1 * Is_2 + (1 - Ss_1) * (1 - Is_1)

    # Generate SNs for threshold
    Ths = np.random.randint(0, 256, Xs.shape, dtype='uint8')
    Ths = np.less(Ths, pth * 256)
    Ths = Ths.astype(np.uint8)

    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            current_state = fsm_absolute_value_init(Xs[i, j])
            for r_idx in range(Xs.shape[2]):
                current_state, Xs[i, j, r_idx] = fsm_absolute_value(current_state, Xs[i, j, r_idx])

    Ys = Ss_2 * Xs + (1 - Ss_2) * (1 - Ths)

    # np.set_printoptions(threshold=np.inf)
    # print(np.amax(np.mean(Ys, axis=-1) * 256))
    # save_image(np.mean(Ys, axis=-1) * 256, outfilename)

    for i in range(Ys.shape[0]):
        for j in range(Ys.shape[1]):
            current_state = fsm_tanh_init(Ys[i, j])
            for r_idx in range(Ys.shape[2]):
                current_state, Os[i, j, r_idx] = fsm_tanh(current_state, Ys[i, j, r_idx])

    save_image(np.mean(Os, axis=-1) * 256, outfilename)
