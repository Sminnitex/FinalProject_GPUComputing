import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("output/timeShared0.csv")
df2 = pd.read_csv("output/timeShared1.csv")
df3 = pd.read_csv("output/timeShared2.csv")
df4 = pd.read_csv("output/timeShared3.csv")
df5 = pd.read_csv("output/timeGlobal0.csv")
df6 = pd.read_csv("output/timeGlobal1.csv")
df7 = pd.read_csv("output/timeGlobal2.csv")
df8 = pd.read_csv("output/timeGlobal3.csv")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(df['Dimensions'], df['Time'], label='blk-grid size 64-14', color='blue')
axes[0].plot(df2['Dimensions'], df2['Time'], label='32-7', color='red')
axes[0].plot(df3['Dimensions'], df3['Time'], label='16-3', color='green')
axes[0].plot(df4['Dimensions'], df4['Time'], label='8-1', color='yellow')
axes[0].set_title('Shared Memory Matrix Transpose')
axes[0].legend()
axes[0].set_xlim(1024, 2047)
axes[0].set_ylim(0, 0.0140)

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

mean = np.mean(df8['Time'])
median = np.median(df8['Time'])
std_dev = np.std(df8['Time'])
min_val = np.min(df8['Time'])
max_val = np.max(df8['Time'])
sum = np.sum(df8['Time'])

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Min:", min_val)
print("Max:", max_val)
print("Sum: ", sum)

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