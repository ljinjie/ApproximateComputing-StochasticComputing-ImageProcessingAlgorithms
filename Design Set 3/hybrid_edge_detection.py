import numpy as np
from scipy import ndimage
from PIL import Image

roberts_cross_v = np.array([[1, 0],
                            [0, -1]])

roberts_cross_h = np.array([[0, 1],
                            [-1, 0]])

stoch_bits = 5
stoch = int(2**stoch_bits)


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

    infilename = 'edge_detection_original.jpg'
    outfilename = 'perm_ED_S3_' + str(stoch_bits) + '_stochastic_bits_subtraction.jpg'

    outfilename_1 = 'output_conv_circle.jpg'
    outfilename_2 = 'output_stoch_circle.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image_conv = (image // stoch) * stoch
    image_stoch = image % stoch

    # conventional part
    vertical = ndimage.convolve(image_conv, roberts_cross_v)
    horizontal = ndimage.convolve(image_conv, roberts_cross_h)

    output_image_conv = abs(vertical) + abs(horizontal)
    output_image_conv = output_image_conv[:-1, :-1]
    save_image(output_image_conv, outfilename_1)

    # stochastic part

    # fixed SN length -- 256
    # Is = np.random.randint(0, 255 % stoch + 1, (image.shape[0] - 1, image.shape[1] - 1, 256), dtype='uint8')
    # changing SN length -- stoch
    Is = np.random.randint(0, 255 % stoch + 1, (image.shape[0] - 1, image.shape[1] - 1, stoch), dtype='uint8')

    Is_1 = (Is < image_stoch[:-1, :-1, np.newaxis])
    Is_2 = (Is < image_stoch[1:, :-1, np.newaxis])
    Is_3 = (Is < image_stoch[:-1, 1:, np.newaxis])
    Is_4 = (Is < image_stoch[1:, 1:, np.newaxis])

    t_1 = np.logical_xor(Is_1, Is_4)
    t_2 = np.logical_xor(Is_2, Is_3)

    # Generate SNs for select lines
    Ss = np.random.randint(0, 2, t_1.shape, dtype='uint8')

    output_image_stoch = t_1 * Ss + t_2 * (1 - Ss)
    output_image_stoch = np.mean(output_image_stoch, axis=-1) * (stoch - 1)
    save_image(output_image_stoch, outfilename_2)

    save_image(output_image_conv - output_image_stoch, outfilename)
