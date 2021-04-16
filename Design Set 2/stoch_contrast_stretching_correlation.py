import numpy as np
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

n = 8 - trunc_bits
pow2n = int(2**n)


def fsm_jk_flip_flop(current_state, j, k):
    if current_state == 0:
        # return 0, 0 if j == 0 else 1, 0
        if j == 0:
            return 0, 0
        else:
            return 1, 0
    else:
        # return 1, 1 if k == 0 else 0, 1
        if k == 0:
            return 1, 1
        else:
            return 0, 1


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
    outfilename = 'CS_S2_correlation_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc

    Is = np.random.randint(0, 256 // trunc, (*image.shape, pow2n), dtype='uint8')
    # Is = np.random.randint(0, 256 // trunc, (*image.shape, 256), dtype='uint8')
    # Is = (Is < image[..., np.newaxis])
    # Is = Is.astype(np.uint8)
    # a = 100 // trunc
    # b = 155 // trunc
    a = np.amin(image)
    b = np.amax(image)
    Is_a = np.less(Is, a)
    Is_b = np.less(Is, b)
    Is_x = (Is < image[..., np.newaxis])

    J = np.logical_and(Is_x, np.logical_not(Is_a))
    K = np.logical_and(np.logical_not(Is_x), Is_b)

    Os = np.zeros(J.shape, dtype='uint8')
    current_state = np.zeros(J.shape, dtype='uint8')

    for i in range(Os.shape[0]):
        for j in range(Os.shape[1]):
            for k in range(Os.shape[2]):
                if k == 0:
                    current_state[i, j, k], Os[i, j, k] = fsm_jk_flip_flop(np.random.randint(0, 2), J[i, j, k],
                                                                           K[i, j, k])
                else:
                    current_state[i, j, k], Os[i, j, k] = fsm_jk_flip_flop(current_state[i, j, k - 1], J[i, j, k],
                                                                           K[i, j, k])

    save_image(np.sum(Os, axis=-1) * trunc, outfilename)
    # save_image(np.sum(Os, axis=-1), outfilename)
