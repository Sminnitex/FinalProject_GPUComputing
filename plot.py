import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df0 = pd.read_csv("output/Matrix0.csv")
df1 = pd.read_csv("output/Matrix1.csv")
df2 = pd.read_csv("output/Matrix2.csv")
df3 = pd.read_csv("output/Matrix3.csv")
df4 = pd.read_csv("output/Matrix4.csv")
df5 = pd.read_csv("output/Matrix5.csv")
df6 = pd.read_csv("output/Matrix6.csv")
df7 = pd.read_csv("output/Matrix7.csv")
df8 = pd.read_csv("output/Matrix8.csv")
#df9 = pd.read_csv("output/Matrix9.csv")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(df['Dimensions'], df['Bandwidth'], label='blk-grid size 64-14', color='blue')
axes[0].plot(df2['Dimensions'], df2['Bandwidth'], label='32-7', color='red')
axes[0].plot(df3['Dimensions'], df3['Bandwidth'], label='16-3', color='green')
axes[0].plot(df4['Dimensions'], df4['Bandwidth'], label='8-1', color='yellow')
axes[0].set_title('Shared Memory Matrix Transpose')
axes[0].legend()

# Plot second graph
axes[1].plot(df5['Dimensions'], df5['Time'], label='64-14', color='blue')
axes[1].plot(df6['Dimensions'], df6['Time'], label='32-7', color='red')
axes[1].plot(df7['Dimensions'], df7['Time'], label='16-3', color='green')
axes[1].plot(df8['Dimensions'], df8['Time'], label='8-1', color='yellow')
axes[1].set_title('Global Memory Matrix Transpose')
axes[1].legend()

plt.xlabel("Matrix n*n dimension")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.xlim(1024, 2047)
plt.ylim(0, 0.0140)
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
bandwidth1 = (34358647168 / np.power(10, 9)) / 1.3184719999999999
bandwidth2 = (34358647168 / np.power(10, 9)) / 0.027733
bandwidth3 = (34358647168 / np.power(10, 9)) / 0.016306
bandwidth4 = (34358647168 / np.power(10, 9)) / 0.013561
bandwidth5 = (34358647168 / np.power(10, 9)) / 0.669816
bandwidth6 = (34358647168 / np.power(10, 9)) / 0.067616
bandwidth7 = (34358647168 / np.power(10, 9)) / 0.017985
bandwidth8 = (34358647168 / np.power(10, 9)) / 0.013659999999999999
firstplot = [bandwidth1, bandwidth2, bandwidth3, bandwidth4]
secondplot = [bandwidth5, bandwidth6, bandwidth7, bandwidth8]

axes[0].plot(["64-14", "32-7", "16-3", "8-1"], firstplot, color='blue')
axes[0].set_title('Shared Memory Matrix Transpose')

# Plot second graph
axes[1].plot(["64-14", "32-7", "16-3", "8-1"], secondplot, color='blue')
axes[1].set_title('Global Memory Matrix Transpose')

plt.xlabel("Block and grid sizes")
plt.ylabel("Bandwidth GB/s")
plt.tight_layout()
plt.show()