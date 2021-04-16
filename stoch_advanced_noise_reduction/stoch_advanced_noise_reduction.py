import numpy as np
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

n = 8
pow2n = int(2**n)

thres = 20


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
    outfilename_1 = 'NR_' + str(trunc_bits) + '_bits_truncated.jpg'
    outfilename_2 = 'NR_advanced_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc

    Is = np.zeros((*image.shape,  pow2n // trunc), dtype='uint8')
    Rs = np.random.permutation(pow2n // trunc)
    for i_idx in range(Is.shape[0]):
        for j_idx in range(Is.shape[1]):
            Is[i_idx, j_idx] = (Rs < image[i_idx, j_idx])

    # Initialize output array
    Os = np.zeros((Is.shape[0] - 2, Is.shape[1] - 2, Is.shape[2]), dtype='uint8')
    out = np.zeros((Is.shape[0] - 2, Is.shape[1] - 2), dtype='uint8')

    for i in range(Os.shape[0]):
        for j in range(Os.shape[1]):
            Os[i, j] = maj_9_input(Is[i:i + 3, j:j + 3])
            s = np.sum(Os[i, j])

            if abs(s - image[i + 1, j + 1]) < thres:
                out[i, j] = image[i + 1, j + 1]
            else:
                out[i, j] = s

    save_image(np.sum(Os, axis=-1), outfilename_1)
    save_image(out, outfilename_2)
