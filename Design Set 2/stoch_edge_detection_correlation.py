import numpy as np
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

n = 8 - trunc_bits
pow2n = int(2**n)


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
    outfilename = 'ED_S2_correlation_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = image // trunc

    # Is = np.random.randint(0, 256 // trunc, (image.shape[0] - 1, image.shape[1] - 1, pow2n), dtype='uint8')
    Is = np.random.randint(0, 256 // trunc, (image.shape[0] - 1, image.shape[1] - 1, 256), dtype='uint8')
    # Is = (Is < image[..., np.newaxis])
    # Is = Is.astype(np.uint8)
    Is_1 = (Is < image[:-1, :-1, np.newaxis])
    Is_2 = (Is < image[1:, :-1, np.newaxis])
    Is_3 = (Is < image[:-1, 1:, np.newaxis])
    Is_4 = (Is < image[1:, 1:, np.newaxis])

    t_1 = np.logical_xor(Is_1, Is_4)
    t_2 = np.logical_xor(Is_2, Is_3)

    # Generate SNs for select lines
    Ss = np.random.randint(0, 2, t_1.shape, dtype='uint8')

    Os = t_1 * Ss + t_2 * (1 - Ss)

    # save_image(np.sum(Os, axis=-1) * trunc, outfilename)
    save_image(np.sum(Os, axis=-1), outfilename)
