import numpy as np
import matplotlib.pyplot as plt
import statistics as st

pow2n = 256
runs = 100

if __name__ == '__main__':
    inputs = np.random.randint(0, pow2n, (runs, 3), dtype='uint8')
    actual_median = np.zeros(runs, dtype='uint8')
    Is = np.zeros((*inputs.shape, pow2n), dtype='uint8')
    Rs = np.random.permutation(pow2n)
    for r_idx in range(runs):
        actual_median[r_idx] = st.median(inputs[r_idx])
        for i_idx in range(inputs.shape[1]):
            Is[r_idx, i_idx] = np.less(Rs, inputs[r_idx, i_idx])

    Is_add = np.sum(Is, axis=1)
    Is_maj = np.greater(Is_add, 1)
    Is_median_output = np.sum(Is_maj, axis=-1)

    x = np.arange(1, runs + 1)

    FONTSIZE = 14
    plt.plot(x, actual_median, 'x', label='ideal')
    plt.plot(x, Is_median_output, '+', label='MAJ outputs')
    plt.xlabel("Runs", fontsize=FONTSIZE)
    plt.ylabel("Ideal vs. MAJ Outputs", fontsize=FONTSIZE)
    plt.title('3-Input MAJ as Median Filter')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
