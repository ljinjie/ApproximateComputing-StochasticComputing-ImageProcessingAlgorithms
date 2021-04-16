# import some important libraries
import numpy as np
import matplotlib.pyplot as plt

'''
The error of a unipolar multiplier (i.e., AND gate) with inputs X and Y and output Z is a function of:
1) p_x, the probability or, equivalently, the unipolar value of X
2) p_y, the probability or, equivalently, the unipolar value of Y
3) N, the SN length
4) n, the bit-width of the SNGs (we will use large enough n so that n won't play a large role)

The measure of error matters. For this tutorial, we'll use mean squared error (MSE). MSE is the average square 
difference between the estimated output value, Z_hat, and the true output value, Z_star. MSE = E[(Z_hat - Z_star)^2].
In our discussions on Bluejeans, I used the word "Variance" to talk about average squared error. For the unipolar
multiplier, variance and MSE are the same thing if n is large.

This tutorial will be an example of a common analysis that you may want to do. This analysis can be done analytically
or via simulation. We will focus on simulation today.
'''

def simulate_single_px_py(px, py, n, runs):
    pow2n = int(2 ** n)  # a useful value to have
    N = pow2n  # SN length
    errors = np.zeros(runs)  # an array that will hold all the errors we record

    Z_star = (px + py) / 2  # target value of multiplication
    # Begin simulation loop
    for r_idx in range(runs):
        Bx = round(px * pow2n)  # Bx is the B input to X's SNG
        By = round(py * pow2n)  # By is the B input to Y's SNG

        # Rx will be a N-element list of random integers between 0 (inclusive) and pow2n (exclusive).
        # Rx will serve as the RNS for X's SNG. Each one of its elements is the R input to X's SNG during a clock cycle
        Rx = np.random.randint(0, pow2n, N)
        Ry = np.random.randint(0, pow2n, N)  # same as Rx but for Y's SNG
        Se = np.random.randint(0, pow2n, N)  # select line

        # Generate SN X and Y by simulating a comparator.
        # X and Y are now N-element numpy arrays where each element is a bit of X or Y
        X = (Rx < Bx)
        Y = (Ry < By)
        S = np.less(Se, pow2n/2)
        S_neg = np.logical_not(S)
        S = S.astype(int)
        S_neg = S_neg.astype(int)

        # Simulate MUX
        Z = S * X + S_neg * Y

        # count up all the 1's in Z and then divide by length of Z. Z_hat this is an estimate for Z's value
        Z_hat = np.mean(Z)

        errors[r_idx] = Z_hat - Z_star # record error

    # Compute mean squared error
    mse = np.mean(np.square(errors))
    return mse


if __name__ == '__main__':
    #  This code only runs when this python file is the one that is run. This code doesn't run if this file is imported.
    #  All code outside of "if __name__ == '__main__':" will always run even when the file is imported.

    #  Begin by declaring parameters of the simulation
    n = 4  # the bit-width or precision of the SNGs. This determines what values the input SNs can take
    runs = 1000  # number of simulation runs

    # Example of simulations where you use fixed input SN values
    px = 0.5
    py = 0.5
    print("MSE with px={} and py={}: {}".format(px, py, simulate_single_px_py(px, py, n, runs)))

    px = 0
    py = 0.5
    print("MSE with px={} and py={}: {}".format(px, py, simulate_single_px_py(px, py, n, runs)))

    px = 0.25
    py = 0.5
    print("MSE with px={} and py={}: {}".format(px, py, simulate_single_px_py(px, py, n, runs)))

    # Notice how error changes based on px and py. Let's systematically vary px and py and make a 2D heatplot
    pow2n = int(2**n)
    pxs = np.arange(0, pow2n+1)  # a numpy array with values [0, 1, 2, ..., pow2n]
    pxs = pxs / pow2n  # a numpy array with values [0, 1/pow2n, 2/pow2n, ..., 1].
    pys = pxs.copy()

    mses = np.zeros((pow2n+1, pow2n+1))  # a 2D list to record the mse of each (px, py) pair

    for px_idx in range(pow2n+1):
        for py_idx in range(pow2n+1):
            px = pxs[px_idx]
            py = pys[py_idx]
            mses[px_idx, py_idx] = simulate_single_px_py(px, py, n, runs)

    # Now to plot
    FONTSIZE = 14
    plot_x, plot_y = np.meshgrid(pxs, pys)
    plt.pcolormesh(plot_x, plot_y, mses, cmap="plasma")
    plt.colorbar().set_label("MSE", rotation=270, labelpad=15, fontsize=FONTSIZE)
    plt.xlabel("$p_x$", fontsize=FONTSIZE)
    plt.ylabel("$p_y$", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()
