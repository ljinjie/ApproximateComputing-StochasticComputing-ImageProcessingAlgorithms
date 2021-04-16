import numpy as np
import math
from PIL import Image


def find_mse(i1, i2, i3):
    se2 = np.square(i2 - i1)
    se3 = np.square(i3 - i1)
    return np.mean(se2), np.mean(se3)


def find_psnr(mse2, mse3, i2, i3):
    return 20 * math.log(np.amax(i2), 10) - 10 * math.log(mse2, 10), 20 * math.log(np.amax(i3), 10) - 10 * math.log(
        mse3, 10)


def find_mean_ae(i1, i2, i3):
    ae2 = np.abs(i2 - i1)
    ae3 = np.abs(i3 - i1)
    return np.mean(ae2), np.mean(ae3)


def find_max_ae(i1, i2, i3):
    ae2 = np.abs(i2 - i1)
    ae3 = np.abs(i3 - i1)
    return np.amax(ae2), np.amax(ae3)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


if __name__ == '__main__':
    in_image1_name = r'Design Set 1 Results\conv_noise_reduction_new\NR_S1_0_bits_truncated.jpg'
    in_image2_name = r'Design Set 2 Results\stoch_noise_reduction_maj_const\NR_S2_1_bits_truncated_new.jpg'
    in_image3_name = r'Design Set 3 Results\hybrid_noise_reduction_new\NR_S3_5_stochastic_bits_new.jpg'

    image1 = load_image(in_image1_name)
    # image1 = image1[..., 0]

    image2 = load_image(in_image2_name)
    # image2 = image2[..., 0]

    image3 = load_image(in_image3_name)
    # image3 = image3[..., 0]

    print('Dynamic SN length in S2:')
    print('\n')
    mse_i2, mse_i3 = find_mse(image1, image2, image3)
    print('\tMSE of pure stochastic: ' + str(mse_i2))
    print('\tMSE of hybrid: ' + str(mse_i3))
    print('\n')
    psnr_i2, psnr_i3 = find_psnr(mse_i2, mse_i3, image2, image3)
    print('\tPSNR of pure stochastic: ' + str(psnr_i2) + "dB")
    print('\tPSNR of hybrid: ' + str(psnr_i3) + "dB")
    print('\n')
    mean_ae_i2, mean_ae_i3 = find_mean_ae(image1, image2, image3)
    print('\tMean absolute error of pure stochastic: ' + str(mean_ae_i2 * 100 / 255) + "%")
    print('\tMean absolute error of hybrid: ' + str(mean_ae_i3 * 100 / 255) + "%")
    print('\n')
    max_ae_i2, max_ae_i3 = find_max_ae(image1, image2, image3)
    print('\tMax absolute error of pure stochastic: ' + str(max_ae_i2 * 100 / 255) + "%")
    print('\tMax absolute error of hybrid: ' + str(max_ae_i3 * 100 / 255) + "%")
