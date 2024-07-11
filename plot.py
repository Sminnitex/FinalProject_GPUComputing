import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df0 = pd.read_csv("output/Matrix0.csv", skiprows=1)
df1 = pd.read_csv("output/Matrix1.csv", skiprows=1)
df2 = pd.read_csv("output/Matrix2.csv", skiprows=1)
df3 = pd.read_csv("output/Matrix3.csv", skiprows=1)
df4 = pd.read_csv("output/Matrix4.csv", skiprows=1)
df5 = pd.read_csv("output/Matrix5.csv", skiprows=1)
df6 = pd.read_csv("output/Matrix6.csv", skiprows=1)
df7 = pd.read_csv("output/Matrix7.csv", skiprows=1)
df8 = pd.read_csv("output/Matrix8.csv", skiprows=1)
df9 = pd.read_csv("output/Matrix9.csv", skiprows=1)

matrixNNZ = np.array([df0["NonZeros"][0], df1["NonZeros"][0], df2["NonZeros"][0], df3["NonZeros"][0], df4["NonZeros"][0], df5["NonZeros"][0], df6["NonZeros"][0], df7["NonZeros"][0], df8["NonZeros"][0], df9["NonZeros"][0]])
cusparseBand = np.array([df0['Bandwidth'][0], df1['Bandwidth'][0], df2['Bandwidth'][0], df3['Bandwidth'][0], df4['Bandwidth'][0], df5['Bandwidth'][0], df6['Bandwidth'][0], df7['Bandwidth'][0], df8['Bandwidth'][0], df9['Bandwidth'][0]])
globalBand = np.array([df0['Bandwidth'][1], df1['Bandwidth'][1], df2['Bandwidth'][1], df3['Bandwidth'][1], df4['Bandwidth'][1], df5['Bandwidth'][1], df6['Bandwidth'][1], df7['Bandwidth'][1], df8['Bandwidth'][1], df9['Bandwidth'][1]])
sharedBand = np.array([df0['Bandwidth'][2], df1['Bandwidth'][2], df2['Bandwidth'][2], df3['Bandwidth'][2], df4['Bandwidth'][2], df5['Bandwidth'][2], df6['Bandwidth'][2], df7['Bandwidth'][2], df8['Bandwidth'][2], df9['Bandwidth'][2]])
mySparseBand = np.array([df0['Bandwidth'][3], df1['Bandwidth'][3], df2['Bandwidth'][3], df3['Bandwidth'][3], df4['Bandwidth'][3], df5['Bandwidth'][3], df6['Bandwidth'][3], df7['Bandwidth'][3], df8['Bandwidth'][3], df9['Bandwidth'][3]])
matrixDIM = np.array([df0["Rows"][0] * df0["Columns"][0], df1["Rows"][0] * df1["Columns"][0], df2["Rows"][0] * df2["Columns"][0], df3["Rows"][0] * df3["Columns"][0], df4["Rows"][0] * df4["Columns"][0], df5["Rows"][0] * df5["Columns"][0], df6["Rows"][0] * df6["Columns"][0], df7["Rows"][0] * df7["Columns"][0], df8["Rows"][0] * df8["Columns"][0], df9["Rows"][0] * df9["Columns"][0]])

#Get the sorted indices
file_indices = np.arange(len(matrixNNZ))
sorted_indices_nnz = np.argsort(matrixNNZ)
sorted_indices_dim = np.argsort(matrixDIM)

#Plot by nonzeros
sorted_matrixNNZ = matrixNNZ[sorted_indices_nnz]
sorted_cusparseBand = cusparseBand[sorted_indices_nnz]
sorted_globalBand = globalBand[sorted_indices_nnz]
sorted_sharedBand = sharedBand[sorted_indices_nnz]
sorted_mySparseBand = mySparseBand[sorted_indices_nnz]

fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))

axes1[0].plot(sorted_matrixNNZ, sorted_cusparseBand, label='Cusparse', color='blue')
axes1[0].plot(sorted_matrixNNZ, sorted_globalBand, label='Global', color='red')
axes1[0].plot(sorted_matrixNNZ, sorted_sharedBand, label='Shared', color='green')
axes1[0].plot(sorted_matrixNNZ, sorted_mySparseBand, label='MySparse', color='yellow')

axes1[1].plot(file_indices, matrixNNZ, label='Nonzeros', color='blue')
axes1[1].set_xticks(file_indices) 
labels = ['1138_bus', 'Maragal_3', 'photogrammetry', 'plbuckle', 'bcsstk17', 'filter2D', 'SiH4', 'linverse', 't2dah_a', 'bcsstk35']
axes1[1].set_xticklabels(labels, rotation=45)
axes1[1].set_xlabel('Files')
axes1[1].set_ylabel('Non zeros')
axes1[1].legend()

axes1[0].set_xlabel('NonZeros')
axes1[0].set_ylabel('Bandwidth')
axes1[0].legend()

#Plot by matrix dim
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
sorted_matrixDIM = matrixDIM[sorted_indices_dim]
sorted_cusparseBand = cusparseBand[sorted_indices_dim]
sorted_globalBand = globalBand[sorted_indices_dim]
sorted_sharedBand = sharedBand[sorted_indices_dim]
sorted_mySparseBand = mySparseBand[sorted_indices_dim]

axes2[0].plot(sorted_matrixDIM, sorted_cusparseBand, label='Cusparse', color='blue')
axes2[0].plot(sorted_matrixDIM, sorted_globalBand, label='Global', color='red')
axes2[0].plot(sorted_matrixDIM, sorted_sharedBand, label='Shared', color='green')
axes2[0].plot(sorted_matrixDIM, sorted_mySparseBand, label='MySparse', color='yellow')

axes2[1].plot(file_indices, matrixDIM, label='Dimensions', color='blue')
axes2[1].set_xticks(file_indices) 
axes2[1].set_xticklabels(labels, rotation=45)
axes2[1].set_xlabel('Files')
axes2[1].set_ylabel('Dimensions')
axes2[1].legend()

axes2[0].set_xlabel('Dimensions')
axes2[0].set_ylabel('Bandwidth')
axes2[0].legend()

# Plot by files
fig3, axes3 = plt.subplots(figsize=(10, 8))

axes3.plot(file_indices, cusparseBand, label='Cusparse', color='blue')
axes3.plot(file_indices, globalBand, label='Global', color='red')
axes3.plot(file_indices, sharedBand, label='Shared', color='green')
axes3.plot(file_indices, mySparseBand, label='MySparse', color='yellow')

axes3.set_xticks(file_indices)
axes3.set_xticklabels(['1138_bus', 'Maragal_3', 'photogrammetry', 'plbuckle', 'bcsstk17', 'filter2D', 'SiH4', 'linverse', 't2dah_a', 'bcsstk17'], rotation=45)
axes3.set_xlabel('Files')
axes3.set_ylabel('Bandwidth')
axes3.legend()

plt.tight_layout()
plt.show()