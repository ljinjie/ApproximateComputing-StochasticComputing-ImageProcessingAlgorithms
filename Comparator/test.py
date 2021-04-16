import pandas as pd

ss_dis = pd.read_excel('16_state_steady_state_distribution.xlsx', header=None)
ss_dis = ss_dis.to_numpy(dtype='uint8')

print(ss_dis[0])