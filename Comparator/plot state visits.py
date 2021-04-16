import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_STATES = 32
x_val = 0

x = np.arange(1, MAX_STATES + 1)
visit_counts = pd.read_excel('State Visit Counts ' + str(MAX_STATES) + ' State_Fixed x' + str(x_val) +
                                                                       ' y0.5.xlsx', header=None)
visit_counts = visit_counts.to_numpy()

FONTSIZE = 14

for i in range(MAX_STATES):
    plt.plot(x, visit_counts[i, :], label=(str(16*(i + 1)) + ' warm-up cycles'))
plt.xlabel("State #", fontsize=FONTSIZE)
plt.ylabel("Visit Counts", fontsize=FONTSIZE)
plt.title(str(MAX_STATES) + '-state FSM State Visit Counts')
plt.tight_layout()
plt.legend(loc='best')
plt.show()
