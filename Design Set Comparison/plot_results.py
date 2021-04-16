import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 8)
MSE_con = [356.73, 368.45, 360.12, 374.26, 501.45, 966.67, 28971.69, 28986.93]
MSE_dyn = [361.88, 511.76, 697.37, 906.27, 1305.49, 2588.10, 33734.93, 32665.64]

FONTSIZE = 14

plt.plot(x, MSE_con, label='Constant SN Length')
plt.plot(x, MSE_dyn, label='Dynamic SN Length')
plt.xlabel("Number of LSBs truncated", fontsize=FONTSIZE)
plt.ylabel("MSE", fontsize=FONTSIZE)
plt.title('MSEs of Outputs with Constant/Dynamic SN Length (Frame Difference)')
plt.tight_layout()
plt.legend(loc='best')
plt.show()
