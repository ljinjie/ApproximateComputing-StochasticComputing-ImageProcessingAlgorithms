import numpy as np
import math
from PIL import Image


def find_mse(i1, i2):
    se2 = np.square(i2 - i1)
    return np.mean(se2)


def find_psnr(mse2, i2):
    return 20 * math.log(np.amax(i2), 10) - 10 * math.log(mse2, 10)


def find_mean_ae(i1, i2):
    ae2 = np.abs(i2 - i1)
    return np.mean(ae2)


def find_max_ae(i1, i2):
    ae2 = np.abs(i2 - i1)
    return np.amax(ae2)


def find_mean_re(i1, i2):
    ae2 = np.abs(i2 - i1)
    ae2 = (ae2 + 1) / (i1 + 1)
    return np.mean(ae2)


def find_max_re(i1, i2):
    ae2 = np.abs(i2 - i1)
    ae2 = (ae2 + 1) / (i1 + 1)
    return np.amax(ae2)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


if __name__ == '__main__':
    in_image1_name = r'Design Set 1 Results\conv_noise_reduction_new\NR_S1_0_bits_truncated.jpg'
    in_image2_name = r'hardware_hybrid_nr_4_4.jpg'

    image1 = load_image(in_image1_name)
    # image1 = image1[..., 0]

    image2 = load_image(in_image2_name)
    # image2 = image2[..., 0]

    mse_i2 = find_mse(image1, image2)
    print('\tMSE: ' + str(mse_i2))
    print('\n')
    psnr_i2 = find_psnr(mse_i2, image2)
    print('\tPSNR: ' + str(psnr_i2) + "dB")
    print('\n')
    mean_ae_i2 = find_mean_ae(image1, image2)
    print('\tMean absolute error: ' + str(mean_ae_i2 * 100 / 255) + "%")
    print('\n')
    max_ae_i2 = find_max_ae(image1, image2)
    print('\tMax absolute error: ' + str(max_ae_i2 * 100 / 255) + "%")
    # print('\n')
    # mean_re_i2, mean_re_i3 = find_mean_re(image1, image2, image3)
    # print('\tMean relative error of pure stochastic: ' + str(mean_re_i2 * 100) + "%")
    # print('\tMean relative error of hybrid: ' + str(mean_re_i3 * 100) + "%")
    # print('\n')
    # max_re_i2, max_re_i3 = find_max_re(image1, image2, image3)
    # print('\tMax relative error of pure stochastic: ' + str(max_re_i2 * 100) + "%")
    # print('\tMax relative error of hybrid: ' + str(max_re_i3 * 100) + "%")
