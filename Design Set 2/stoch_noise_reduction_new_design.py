import numpy as np
from PIL import Image

trunc_bits = 7
trunc = int(2**trunc_bits)

n = 8
pow2n = int(2**n)


def maj_9_input(Is_partial):
    Is_add = np.sum(np.sum(Is_partial, axis=0), axis=0)
    return np.greater(Is_add, 4)


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
    outfilename = 'NR_S2_' + str(trunc_bits) + '_bits_truncated_new.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc * trunc

    Is = np.zeros((*image.shape,  pow2n), dtype='uint8')
    Rs = np.random.permutation(pow2n)
    for i_idx in range(Is.shape[0]):
        for j_idx in range(Is.shape[0]):
            Is[i_idx, j_idx] = (Rs < image[i_idx, j_idx])

    # Initialize output array
    Os = np.zeros((Is.shape[0] - 2, Is.shape[1] - 2, Is.shape[2]), dtype='uint8')

    for i in range(Os.shape[0]):
        print('Working on pixel row ' + str(i))
        for j in range(Os.shape[1]):
            Os[i, j] = maj_9_input(Is[i:i + 3, j:j + 3])

    save_image(np.mean(Os, axis=-1) * 256, outfilename)
