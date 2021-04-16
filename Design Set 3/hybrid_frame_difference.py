import numpy as np
from PIL import Image
import pandas as pd

stoch_bits = 3
stoch = int(2**stoch_bits)
MAX_STATES = 16

# frame difference threshold parameters
thres = 40


def fsm_tanh(current_state, x):
    if x == 1 and current_state != MAX_STATES - 1:
        if current_state < MAX_STATES // 2:
            return current_state + 1, 0
        else:
            return current_state + 1, 1
    elif x == 1 and current_state == MAX_STATES - 1:
        return current_state, 1
    elif x == 0 and current_state != 0:
        if current_state < MAX_STATES // 2:
            return current_state - 1, 0
        else:
            return current_state - 1, 1
    elif x == 0 and current_state == 0:
        return current_state, 0


def fsm_tanh_init(runs, ts):
    current_state = np.random.randint(0, MAX_STATES)
    for init_idx in range(runs):
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
    return 0 if diff < (thres // stoch * stoch) else 255


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    infilename_1 = 'frame_difference_original_1_small.jpg'
    infilename_2 = 'frame_difference_original_2_small.jpg'
    outfilename = 'FD_S3_' + str(stoch_bits) + '_stochastic_bits.jpg'

    image_1 = load_image(infilename_1)
    image_1 = image_1[..., 0]
    image_1_conv = (image_1 // stoch) * stoch
    image_1_stoch = image_1 % stoch

    image_2 = load_image(infilename_2)
    image_2 = image_2[..., 0]
    image_2_conv = (image_2 // stoch) * stoch
    image_2_stoch = image_2 % stoch

    # conventional part
    output_image_conv = np.zeros(image_1.shape)

    for i in range(output_image_conv.shape[0]):
        for j in range(output_image_conv.shape[1]):
            output_image_conv[i, j] = frame_diff(image_1_conv[i, j], image_2_conv[i, j])

    # stochastic part
    ss_dis = pd.read_excel('16_state_steady_state_distribution.xlsx', header=None)
    ss_dis = ss_dis.to_numpy()

    Is = np.random.randint(0, 255 % stoch + 1, (*image_1.shape, 1024), dtype='uint8')
    Is_1 = (Is < image_1_stoch[..., np.newaxis])
    Is_1 = Is_1.astype(np.uint8)
    Is_2 = (Is < image_2_stoch[..., np.newaxis])
    Is_2 = Is_2.astype(np.uint8)

    # Initialize output array
    output_image_stoch = np.zeros(Is.shape, dtype='uint8')

    # Generate SNs for select lines
    Ss = np.random.randint(0, 2, Is.shape, dtype='uint8')
    Ss = np.equal(Ss, 1)
    Ss = Ss.astype(np.uint8)

    Xs = np.logical_xor(Is_1, Is_2)

    # Generate SNs for threshold
    Ths = np.random.randint(0, 255 % stoch + 1, Xs.shape, dtype='uint8')
    Ths = np.less(Ths, thres % stoch)
    Ths = Ths.astype(np.uint8)

    Ys = Ss * Xs + (1 - Ss) * (1 - Ths)

    for i in range(Ys.shape[0]):
        for j in range(Ys.shape[1]):
            est_val = int(maj(Ys[i, j]))
            fsm_choice = ss_dis[est_val]
            set_number = 0
            for ss_idx in range(16):
                for f_idx in range(fsm_choice[ss_idx]):
                    current_state = ss_idx
                    for k in range(32):
                        current_state, output_image_stoch[i, j, set_number * 32 + k] = fsm_tanh(current_state,
                                                                                Ys[i, j, set_number * 32 + k])
                    set_number = set_number + 1

    output_image_stoch = np.mean(output_image_stoch, axis=-1) * (stoch - 1)

    save_image(output_image_conv + output_image_stoch, outfilename)
