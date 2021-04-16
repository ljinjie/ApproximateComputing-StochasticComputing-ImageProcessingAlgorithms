import numpy as np
from scipy import ndimage
from PIL import Image

trunc_bits = 0
trunc = int(2**trunc_bits)

# frame difference threshold parameters
thres = 60


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def frame_diff(f1, f2):
    diff = abs(f1 - f2)
    return 0 if diff < thres else 255


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    # infilename_1 = 'frame_difference_original_1_small.jpg'
    # infilename_2 = 'frame_difference_original_2_small.jpg'
    infilename_1 = '1_1.png'
    infilename_2 = '2_2.png'
    outfilename = 'FD_S1_' + str(trunc_bits) + '_bits_truncated.jpg'

    image_1 = load_image(infilename_1)
    image_1 = image_1[..., 0]
    image_1 = (image_1 // trunc) * trunc
    print(image_1[0][0])

    image_2 = load_image(infilename_2)
    image_2 = image_2[..., 0]
    image_2 = (image_2 // trunc) * trunc
    print(image_2[0][0])

    output_image = np.zeros(image_1.shape)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = frame_diff(image_1[i, j], image_2[i, j])

    # save_image(output_image, outfilename)
