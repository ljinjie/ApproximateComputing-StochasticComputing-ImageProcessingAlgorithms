# import some important libraries
import numpy as np
import matplotlib.pyplot as plt

'''
The error of a unipolar multiplier (i.e., AND gate) with inputs X and Y and output Z is a function of:
1) p_x, the probability or, equivalently, the unipolar value of X
2) p_y, the probability or, equivalently, the unipolar value of Y
3) N, the SN length
4) n, the bit-width of the SNGs (we will use large enough n so that n won't play a large role)

The measure of error matters. For this tutorial, we'll use mean squared error (MSE). MSE if the average square 
difference between the estimated output value, Z_hat, and the target output value, Z_star. MSE = E[(Z_hat - Z_star)^2].
In our discussions on Bluejeans, I used the word "Variance" to talk about average squared error. For the unipolar
multiplier, variance and MSE are the same thing if n is large.

This tutorial will be the same as the last, but the code will be more efficient.
'''

if __name__ == '__main__':
    # Replacing large for loops with built-in or library functions is a good way to speed up code.
    # We will make the same plot as last time
    n = 4
    runs = 1000
    pow2n = int(2**n)
    N = pow2n  # SN length

    pxs = np.arange(pow2n+1)/pow2n
    pxs = 2*pxs-1

    # pxs and pys are 2D arrays where all possible pairings of elements from pxs and elements from pxs appear
    # uncomment the print statements to get a clearer idea of what these two arrays are
    pxs, pys = np.meshgrid(pxs, pxs)
    # print(pxs)
    # print(pys)

    # SNGs take in integers that relate to unipolar value in the following way: p = B / pow2n
    # Bxs = (pxs + 1) * pow2n / 2
    # Bys = (pys + 1) * pow2n / 2
    Bxs = pxs * pow2n
    Bys = pys * pow2n

    # [:,:,np.newaxis] takes Bxs from shape (17, 17) and makes its shape (17, 17, 1). This necessary or comparison with
    # R later will not work
    Bxs = Bxs[:, :, np.newaxis]

    # "..." can be used to replace one set of consecutive ":"'s when indexing an array. so the following statement is
    # equivalent to the last
    Bys = Bys[..., np.newaxis]

    # Some other useful variables. Notice how we calculate as much as possible outside of the for loop. and notice
    # how there is 2 fewer for loops compared to last simulation.
    R_shape = (2, *Bxs.shape, N)
    Z_stars = pxs*pys
    errors = np.zeros((len(pxs), len(pys), runs))
    for r_idx in range(runs):
        # generates an array with shape "R_shape" and where each element is a random int in [0, pow2n)
        Rs = np.random.randint(-pow2n, pow2n, R_shape)
        # Rs = np.random.randint(0, pow2n, R_shape)

        Xs = Rs[0] < Bxs[..., np.newaxis]  # generate many X SNs.
        Ys = Rs[1] < Bys[..., np.newaxis]  # generate many Y SNs

        Zs = np.logical_not(np.logical_xor(Xs, Ys))  # process the many X SNs and Y SNs through AND gate logic

        Z_hats = 2*(np.mean(Zs.astype(int), axis=-1)[..., 0])-1  # count the bits over the last dimension of the Zs array

        errors[..., r_idx] = Z_hats - Z_stars

    mses = np.mean(np.square(errors), axis=-1)
    # Now to plot
    FONTSIZE = 14
    plt.pcolormesh(pxs, pys, mses, cmap="plasma")
    plt.colorbar().set_label("MSE", rotation=270, labelpad=15, fontsize=FONTSIZE)
    plt.xlabel("$X$", fontsize=FONTSIZE)
    plt.ylabel("$Y$", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()
