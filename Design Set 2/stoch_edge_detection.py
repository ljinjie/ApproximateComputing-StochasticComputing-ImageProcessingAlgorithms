import numpy as np
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

n = 10 - trunc_bits
pow2n = int(2**n)

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


def fsm_absolute_value_init(i, j, Xs):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(MAX_STATES):
        current_state, y = fsm_absolute_value(current_state, Xs[i, j, init_idx])
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

    infilename = 'edge_detection_original.jpg'
    outfilename = 'ED_S2_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc

    # Fixed clock cycles
    Is = np.random.randint(0, 256 / trunc, (*image.shape, 1024), dtype='uint8')
    # Changing clock cycles
    # Is = np.random.randint(0, 256 / trunc, (*image.shape, pow2n), dtype='uint8')
    Is = (Is < image[..., np.newaxis])
    Is_neg = np.logical_not(Is)
    Is = Is.astype(np.uint8)
    Is_neg = Is_neg.astype(np.uint8)

    # Initialize output array
    Os = np.zeros((Is.shape[0] - 1, Is.shape[1] - 1, Is.shape[2]), dtype='uint8')

    # Generate SNs for select lines
    Ss = np.random.randint(0, 2, (3, *Os.shape), dtype='uint8')

    t_1 = (1 - Ss[0]) * Is[:-1, :-1] + Ss[0] * Is_neg[1:, 1:]
    t_2 = (1 - Ss[1]) * Is[:-1, 1:] + Ss[1] * Is_neg[1:, :-1]

    for i in range(t_1.shape[0]):
        for j in range(t_1.shape[1]):
            current_state = fsm_absolute_value_init(i, j, t_1)
            for r_idx in range(t_1.shape[2]):
                current_state, t_1[i, j, r_idx] = fsm_absolute_value(current_state, t_1[i, j, r_idx])

    for i in range(t_2.shape[0]):
        for j in range(t_2.shape[1]):
            current_state = fsm_absolute_value_init(i, j, t_2)
            for r_idx in range(t_2.shape[2]):
                current_state, t_2[i, j, r_idx] = fsm_absolute_value(current_state, t_2[i, j, r_idx])

    Os = (1 - Ss[2]) * t_1 + Ss[2] * t_2

    save_image((np.mean(Os, axis=-1) - 0.5) * 512, outfilename)
