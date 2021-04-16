import numpy as np
from PIL import Image

stoch_bits = 3
stoch = int(2**stoch_bits)
MAX_STATES = 16

# frame difference threshold parameters
thres = 40


def fsm_tanh(current_state, x):
    if x == 1 and current_state != MAX_STATES - 1:
        current_state = current_state + 1
    elif x == 0 and current_state != 0:
        current_state = current_state - 1

    if current_state < MAX_STATES // 2:
        return current_state, 0
    else:
        return current_state, 1


def fsm_tanh_init(ts):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(MAX_STATES * 8):
        current_state, y = fsm_tanh(current_state, ts[init_idx % ts.size])
    return current_state


def maj(ts):
    outputs = np.zeros(32)
    for i in range(32):
        sum = 0
        for j in range(32):
            sum = sum + ts[i + j * 32]
        outputs[i] = 1 if sum > 15 else 0
    return np.sum(outputs)


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
    return 15 if diff > 2 else 0


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    infilename_1 = 'frame_difference_original_1_small.jpg'
    infilename_2 = 'frame_difference_original_2_small.jpg'
    # infilename_1 = '1_1.png'
    # infilename_2 = '2_2.png'
    outfilename = 'perm_FD_S3_' + str(stoch_bits) + '_stochastic_bits.jpg'

    image_1 = load_image(infilename_1)
    image_1 = image_1[..., 0]
    image_1_conv = (image_1 // stoch)
    image_1_stoch = image_1 % stoch

    image_2 = load_image(infilename_2)
    image_2 = image_2[..., 0]
    image_2_conv = (image_2 // stoch)
    image_2_stoch = image_2 % stoch

    # conventional part
    output_image_conv = np.zeros(image_1.shape)

    for i in range(output_image_conv.shape[0]):
        for j in range(output_image_conv.shape[1]):
            output_image_conv[i, j] = frame_diff(image_1_conv[i, j], image_2_conv[i, j])

    # stochastic part
    Is = np.zeros((*image_1.shape, 256), dtype='uint8')
    for i in range(Is.shape[0]):
        for j in range(Is.shape[1]):
            Is[i, j] = np.random.permutation(256)

    Is_1 = (Is < (image_1_stoch[..., np.newaxis] * 16))
    # Is_1 = Is_1.astype(np.uint8)
    Is_2 = (Is < (image_2_stoch[..., np.newaxis] * 16))
    # Is_2 = Is_2.astype(np.uint8)

    # Initialize output array
    output_image_stoch = np.zeros(Is.shape, dtype='uint8')

    # Generate SNs for select lines
    Ss = np.zeros((*image_1.shape, 256), dtype='uint8')
    for i in range(Ss.shape[0]):
        for j in range(Ss.shape[1]):
            Ss[i, j] = np.random.permutation(256)
    Ss = (Ss < 128)

    Xs = np.logical_xor(Is_1, Is_2)

    # Generate SNs for threshold
    Ths = np.zeros(Xs.shape, dtype='uint8')
    for i in range(Ths.shape[0]):
        for j in range(Ths.shape[1]):
            Ths[i, j] = np.random.permutation(256)
    Ths = (Ths < 128)

    Ys = Ss * Xs + (1 - Ss) * (1 - Ths)

    # current_state = fsm_tanh_init(Ys[0, 0])

    for i in range(Ys.shape[0]):
        for j in range(Ys.shape[1]):
            current_state = fsm_tanh_init(Ys[i, j])
            for r_idx in range(Ys.shape[2]):
                current_state, output_image_stoch[i, j, r_idx] = fsm_tanh(current_state, Ys[i, j, r_idx])

    output_image_stoch = np.sum(output_image_stoch, axis=-1) % 16

    save_image(output_image_conv * 16 + output_image_stoch, outfilename)
