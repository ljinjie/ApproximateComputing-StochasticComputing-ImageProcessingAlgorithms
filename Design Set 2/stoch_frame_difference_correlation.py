import numpy as np
from PIL import Image

trunc_bits = 1
trunc = int(2**trunc_bits)

n = 8 - trunc_bits
pow2n = int(2**n)

# MAX_STATES = 64 if trunc_bits < 5 else 32
MAX_STATES = 64

# pow2n = max(pow2n, MAX_STATES)

# Threshold parameter
th = 40


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
    for init_idx in range(min(MAX_STATES, pow2n)):
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
    outfilename = 'FD_S2_correlation_' + str(trunc_bits) + '_bits_truncated.jpg'
    image_1 = load_image(infilename_1)
    image_1 = image_1[..., 0]
    image_1 = image_1 // trunc

    image_2 = load_image(infilename_2)
    image_2 = image_2[..., 0]
    image_2 = image_2 // trunc

    # Is = np.random.randint(0, 256 // trunc, (*image_1.shape, pow2n), dtype='uint8')
    Is = np.random.randint(0, 256 // trunc, (*image_1.shape, 256), dtype='uint8')
    Is_1 = (Is < image_1[..., np.newaxis])
    Is_1 = Is_1.astype(np.uint8)
    Is_2 = (Is < image_2[..., np.newaxis])
    Is_2 = Is_2.astype(np.uint8)

    # Initialize output array
    Os = np.zeros(Is.shape, dtype='uint8')

    # Generate SNs for select lines
    # Ss = np.random.randint(0, 2, (*image_1.shape, pow2n), dtype='uint8')
    Ss = np.random.randint(0, 2, (*image_1.shape, 256), dtype='uint8')
    Ss = np.equal(Ss, 1)
    Ss = Ss.astype(np.uint8)

    Xs = np.logical_xor(Is_1, Is_2)

    # np.set_printoptions(threshold=np.inf)
    # print(np.sum(Xs, axis=-1))

    # Generate SNs for threshold
    Ths = np.random.randint(0, 256 // trunc, Xs.shape, dtype='uint8')
    Ths = np.less(Ths, th // trunc)
    Ths = Ths.astype(np.uint8)

    Ys = Ss * Xs + (1 - Ss) * (1 - Ths)

    current_state = fsm_tanh_init(Ys[0, 0])

    for i in range(Ys.shape[0]):
        for j in range(Ys.shape[1]):
            # current_state = fsm_tanh_init(Ys[i, j])
            for r_idx in range(Ys.shape[2]):
                current_state, Os[i, j, r_idx] = fsm_tanh(current_state, Ys[i, j, r_idx])

    # save_image(np.sum(Os, axis=-1) * trunc, outfilename)
    save_image(np.sum(Os, axis=-1), outfilename)
