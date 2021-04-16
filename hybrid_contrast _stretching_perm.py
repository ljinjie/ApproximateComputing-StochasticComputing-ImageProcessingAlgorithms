import numpy as np
from PIL import Image

stoch_bits = 4
stoch = int(2**stoch_bits)


def fsm_jk_flip_flop(current_state, j, k):
    if current_state == 0:
        # return 0, 0 if j == 0 else 1, 0
        if j == 0:
            return 0, 0
        else:
            return 1, 0
    else:
        # return 1, 1 if k == 0 else 0, 1
        if k == 0:
            return 1, 1
        else:
            return 0, 1


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def linear_gain(r, c, d):
    return (r - c) * (255 // stoch * stoch) / (d - c)


if __name__ == '__main__':

    infilename = 'contrast_stretching_original.jpg'
    outfilename = 'perm_CS_S3_' + str(stoch_bits) + '_stochastic_bits.jpg'
    image = load_image(infilename)
    image = image[..., 0]
    image_conv = (image // stoch) * stoch
    image_stoch = image % stoch

    # conventional part
    c = np.amin(image_conv)
    d = np.amax(image_conv)
    output_image_conv = np.zeros(image.shape)

    for i in range(output_image_conv.shape[0]):
        for j in range(output_image_conv.shape[1]):
            output_image_conv[i, j] = linear_gain(image_conv[i, j], c, d)

    # stochastic part

    # image SN generation
    Is = np.zeros((*image.shape, stoch), dtype='uint8')
    for i in range(Is.shape[0]):
        for j in range(Is.shape[1]):
            Is[i, j] = np.random.permutation(stoch)

    a = np.amin(image_stoch)
    b = np.amax(image_stoch)
    Is_a = np.less(Is, a)
    Is_b = np.less(Is, b)
    Is_x = (Is < image_stoch[..., np.newaxis])

    J = np.logical_and(Is_x, np.logical_not(Is_a))
    K = np.logical_and(np.logical_not(Is_x), Is_b)

    output_image_stoch = np.zeros(J.shape, dtype='uint8')
    current_state = np.zeros(J.shape, dtype='uint8')

    for i in range(output_image_stoch.shape[0]):
        for j in range(output_image_stoch.shape[1]):
            for k in range(output_image_stoch.shape[2]):
                if k == 0:
                    current_state[i, j, k], output_image_stoch[i, j, k] = fsm_jk_flip_flop(np.random.randint(0, 2),
                                                                                           J[i, j, k],
                                                                                           K[i, j, k])
                else:
                    current_state[i, j, k], output_image_stoch[i, j, k] = fsm_jk_flip_flop(current_state[i, j, k - 1],
                                                                                           J[i, j, k],
                                                                                           K[i, j, k])
    output_image_stoch = np.mean(output_image_stoch, axis=-1) * (stoch - 1)

    save_image(output_image_conv + output_image_stoch, outfilename)
