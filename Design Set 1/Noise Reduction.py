import numpy as np
import statistics as st
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)


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

    infilename = 'nr_poisson.jpg'
    outfilename = 'nr_poisson_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = (image // trunc) * trunc

    print(image[2][2])

    output_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2))

    for m in range(image.shape[0] - 2):
        for n in range(image.shape[1] - 2):
            output_image[m, n] = st.median([image[m, n], image[m, n + 1], image[m, n + 2],
                                            image[m + 1, n], image[m + 1, n + 1],
                                            image[m + 1, n + 2],
                                            image[m + 2, n], image[m + 2, n + 1],
                                            image[m + 2, n + 2]])

    # print(output_image)

    # save_image(output_image, outfilename)
