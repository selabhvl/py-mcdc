# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tcasii import tcas, tcas_num_cond, tcas_names


# Initialize the lists for X and Y
fig, ax = plt.subplots()

data1 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-LongestBool.11-5040-6.csv',header=0, delimiter=',', index_col=0))
data2 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-LongestPath.11-5040-6.csv',header=0, delimiter=',', index_col=0))
data3 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-LongestBoolMay.11-5040-6.csv',header=0, delimiter=',', index_col=0))
data4 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-LongestMayMerge.11-5040-6.csv',header=0, delimiter=',', index_col=0))
data5 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-LongestBetterSize.11-5040-6.csv',header=0, delimiter=',', index_col=0))
data6 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-RandomReuser.11-5040-6.csv',header=0, delimiter=',', index_col=0))
data7 = pd.DataFrame(pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-LongestBetterSize2.11-5040-6.csv',header=0, delimiter=',', index_col=0))
#data = pd.read_csv('~/COEMS/py-mcdc/mcdc_test/pymcdc-5k-6/VS-RandomReuser.11-5040-6.csv',header=0, delimiter=',', index_col=0)

#df1 = pd.DataFrame(data1)
#df2 = pd.DataFrame(data2)
#skiprows = 1
df1 = data1.T #transposed_df
df2 = data2.T
df3 = data3.T
df4 = data4.T
df5 = data5.T
df6 = data6.T
df7 = data7.T

df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3.dropna(inplace=True)
df4.dropna(inplace=True)
df5.dropna(inplace=True)
df6.dropna(inplace=True)
df7.dropna(inplace=True)


LongestBool = list(df1.iloc[:, 0] * 100)
LongestBool1 = list(df1.iloc[:, 1] * 100)
LongestBool2 = list(df1.iloc[:, 2] * 100)
LongestBool3 = list(df1.iloc[:, 3] * 100)

Longest = list(df2.iloc[:, 0] * 100)
Longest1 = list(df2.iloc[:, 1] * 100)
Longest2 = list(df2.iloc[:, 2] * 100)
Longest3 = list(df2.iloc[:, 3] * 100)

LongestBoolMay = list(df3.iloc[:, 0] * 100)
LongestBoolMay1 = list(df3.iloc[:, 1] * 100)
LongestBoolMay2 = list(df3.iloc[:, 2] * 100)
LongestBoolMay3 = list(df3.iloc[:, 3] * 100)

LongestMayMerge = list(df4.iloc[:, 0] * 100)
LongestMayMerge1 = list(df4.iloc[:, 1] * 100)
LongestMayMerge2 = list(df4.iloc[:, 2] * 100)
LongestMayMerge3 = list(df4.iloc[:, 3] * 100)

LongestBetterSize = list(df5.iloc[:, 0] * 100)
LongestBetterSize1 = list(df5.iloc[:, 1] * 100)
LongestBetterSize2 = list(df5.iloc[:, 2] * 100)
LongestBetterSize3 = list(df5.iloc[:, 3] * 100)

LongestBetter2Size = list(df7.iloc[:, 0] * 100)
LongestBetter2Size1 = list(df7.iloc[:, 1] * 100)
LongestBetter2Size2 = list(df7.iloc[:, 2] * 100)
LongestBetter2Size3 = list(df7.iloc[:, 3] * 100)

RandomReuser = list(df6.iloc[:, 0] * 100)
RandomReuser1 = list(df6.iloc[:, 1] * 100)
RandomReuser2 = list(df6.iloc[:, 2] * 100)
RandomReuser3 = list(df6.iloc[:, 3] * 100)

w = 0.12
# Plot the data using bar() method
X = np.arange(1,21)  # the label locations
labels =dict(zip(tcas_names,tcas_num_cond))
#_X = np.arange(len(labels))
plt.bar(X - 3 * w, LongestBool1, w, label=r'$\mathcal{H}_{LPB}$', color='c')
plt.bar(X - 2 * w, Longest1, w, label=r'$\mathcal{H}_{LPN}$', color='m')
plt.bar(X - w, LongestBoolMay1, w, label=r'$\mathcal{H}_{LMMB}$', color='y')
plt.bar(X, LongestMayMerge1, w,  label=r'$\mathcal{H}_{LMMN}$', color='g')
#plt.bar(X + w, LongestBetterSize, w, label='LPBS1', color='dodgerblue')
plt.bar(X +  w, LongestBetter2Size1, w, label=r'$\mathcal{H}_{LPBS}$', color='r')
plt.bar(X + 2 * w, RandomReuser1, w, label=r'$\mathcal{H}_{RR}$',color='k')
        

ax.set_title('% of n+2 solutions, permutations:5040, Runs:6')
ax.set_ylabel('Percentage of n+2 TCs generated (%)')
ax.set_xlabel('TCASII decisions')
ax.set_xticks(X)
ax.set_xticklabels(labels.keys(), rotation = 'vertical', fontsize=8)
ax.legend()
plt.savefig('pymcdc-5k-6/RESULT/Compareheuristics5040-6-2.png', dpi=150)
plt.show()

# Show the plot
plt.show()