import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 8)
MAE_con = [0.74, 0.76, 0.75, 0.77, 1.09, 1.94, 50.60, 50.74]
MAE_dyn = [0.74, 0.96, 1.24, 1.54, 2.20, 4.83, 54.25, 52.11]

FONTSIZE = 14

plt.plot(x, MAE_con, label='Constant SN Length')
plt.plot(x, MAE_dyn, label='Dynamic SN Length')
plt.xlabel("Number of LSBs truncated", fontsize=FONTSIZE)
plt.ylabel("Mean absolute error (%)", fontsize=FONTSIZE)
plt.title('MAEs of Outputs with Constant/Dynamic SN Length (Noise Reduction)')
plt.tight_layout()
plt.legend(loc='best')
plt.show()
