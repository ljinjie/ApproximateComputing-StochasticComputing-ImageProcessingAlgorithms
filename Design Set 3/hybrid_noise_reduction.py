import numpy as np
import statistics as st
from PIL import Image

stoch_bits = 5
stoch = int(2**stoch_bits)
pow2n = stoch


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
    np.set_printoptions(threshold=np.inf)

    infilename = 'noise_reduction_original.jpg'
    outfilename = 'NR_S3_' + str(stoch_bits) + '_stochastic_bits_new.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image_conv = (image // stoch) * stoch
    image_stoch = image % stoch

    # conventional part
    output_image_conv = np.zeros((image.shape[0] - 2, image.shape[1] - 2))

    for m in range(image.shape[0] - 2):
        for n in range(image.shape[1] - 2):
            output_image_conv[m, n] = st.median([image_conv[m, n], image_conv[m, n + 1], image_conv[m, n + 2],
                                                image_conv[m + 1, n], image_conv[m + 1, n + 1], image_conv[m + 1, n + 2],
                                                image_conv[m + 2, n], image_conv[m + 2, n + 1], image_conv[m + 2, n + 2]])
            # output_image_conv[m, n] = median_sorting(image_conv[m, n], image_conv[m, n + 1], image_conv[m, n + 2],
            #                                     image_conv[m + 1, n], image_conv[m + 1, n + 1], image_conv[m + 1, n + 2],
            #                                     image_conv[m + 2, n], image_conv[m + 2, n + 1], image_conv[m + 2, n + 2])

    # print(output_image_conv)

    # stochastic part
    Is = np.zeros((*image.shape,  pow2n), dtype='uint8')
    Rs = np.random.permutation(pow2n)

    for i_idx in range(Is.shape[0]):
        for j_idx in range(Is.shape[0]):
            Is[i_idx, j_idx] = np.less(Rs, image_stoch[i_idx, j_idx])

    # Initialize output array
    output_image_stoch = np.zeros((Is.shape[0] - 2, Is.shape[1] - 2, Is.shape[2]), dtype='uint8')

    for i in range(output_image_stoch.shape[0]):
        for j in range(output_image_stoch.shape[1]):
            output_image_stoch[i, j] = maj_9_input(Is[i:i + 3, j:j + 3])

    output_image_stoch = np.sum(output_image_stoch, axis=-1)
    # output_image_stoch = np.mean(output_image_stoch, axis=-1) * stoch
    # print(output_image_stoch)
    save_image(output_image_conv + output_image_stoch, outfilename)
