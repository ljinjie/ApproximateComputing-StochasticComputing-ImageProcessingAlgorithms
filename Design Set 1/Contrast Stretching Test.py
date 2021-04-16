import numpy as np
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


def linear_gain(r, c, d):
    if r <= c:
        return 0
    elif r >= d:
        return 255
    else:
        return 255 * (r - c) / (d - c)


def linear_gain_part(r, c, d):
    if r <= c:
        return 0
    elif r >= d:
        return 15
    else:
        return 15 * (r - c) / (d - c)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    infilename = 'contrast_stretching_original_408.jpg'
    outfilename = 'CS_S1_' + str(trunc_bits) + '_bits_truncated.jpg'

    outfilename_2 = 'CS_S1_' + str(trunc_bits) + '_bits_truncated_ideal_hybrid.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image = (image // trunc) * trunc
    c = np.amin(image)
    d = np.amax(image)

    output_image = np.zeros(image.shape)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = linear_gain(image[i, j], c, d)

    save_image(output_image, outfilename)
    # print(output_image)

    image_conv = image // 16
    image_stoc = image % 16
    output_image_1 = np.zeros(image.shape)
    output_image_2 = np.zeros(image.shape)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image_1[i, j] = linear_gain_part(image_conv[i, j], np.amin(image_conv), np.amax(image_conv))
            output_image_2[i, j] = linear_gain_part(image_stoc[i, j], np.amin(image_stoc), np.amax(image_stoc))

    # print(output_image_1)
    print(abs(output_image_1 * 16 + output_image_2 - output_image))
    save_image(output_image_1 * 16 + output_image_2, outfilename_2)
