import numpy as np
from PIL import Image

trunc_bits = 7
trunc = int(2**trunc_bits)

# linear gain control parameters
a = 100
b = 155
c = 255 * a / (b - a)
d = 255 / (b - a)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def linear_gain(r):
    s0 = 1 if r < a else 0
    s1 = 1 if r < b else 0

    if s0 == 1 and s1 == 1:
        return 0
    elif s0 == 0 and s1 == 0:
        return 255
    else:
        # return d * (r - c)
        return 128 + 255 * (r - 0.5 * (a + b)) / (b - a)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    infilename = 'contrast_stretching_original_408.jpg'
    outfilename = 'CS_S1_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = (image // trunc) * trunc

    output_image = np.zeros(image.shape)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = linear_gain(image[i, j])

    save_image(output_image, outfilename)
