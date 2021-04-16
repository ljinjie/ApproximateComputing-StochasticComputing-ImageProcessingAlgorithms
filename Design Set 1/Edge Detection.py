import numpy as np
from scipy import ndimage
from PIL import Image

roberts_cross_v = np.array([[1, 0],
                            [0, -1]])

roberts_cross_h = np.array([[0, 1],
                            [-1, 0]])

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


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    infilename = 'edge_detection_original_2.jpg'
    outfilename = 'ED_S1_' + str(trunc_bits) + '_bits_truncated.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = (image // trunc) * trunc
    print(image)

    vertical = ndimage.convolve(image, roberts_cross_v)
    horizontal = ndimage.convolve(image, roberts_cross_h)

    # output_image = np.sqrt(np.square(horizontal) + np.square(vertical))
    output_image = abs(vertical) + abs(horizontal)

    # save_image(output_image, outfilename)
