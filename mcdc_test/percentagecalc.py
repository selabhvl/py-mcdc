
import ast
import numpy as np
import pandas as pd
from math import factorial
import matplotlib.pyplot as plt
import sys
from tcasii import tcas, tcas_num_cond, tcas_names
import csv
import matplotlib.pyplot as bar


ResultMap2 = dict()
fig, ax = plt.subplots()
x = np.arange(1,21)  # the label locations
labels =dict(zip(tcas_names,tcas_num_cond))
PercLists = []
with open('NEWRESULT/H0.42-1000_resultMap.csv', newline='') as csvfile, open('RESULT/Resultoutput.csv', 'w', newline='') as f_output:
    result_map_reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_ResultMap_output = csv.writer(f_output, quoting=csv.QUOTE_NONE)
    #df1 = pd.DataFrame(f_output)
    #df1.head()
    for row in result_map_reader:
        col, rm = int(row[0]), ast.literal_eval(row[1])
        ResultMap2[col] = rm
        #print(rm)

        sum = 0
        Jlist = []
        PercList = []
        for i, j in rm:
            sum = sum + j
            Jlist.append(j)
            counter=0
        for j in Jlist:
            counter = counter+1
            Perc = (100*j)/sum
            print('j={} sum={} Perc={}, counter={}'.format(j, sum, Perc, counter))
            PercList.append(Perc)
        PercLists.append(PercList)
        print(PercLists)
        #fields = ['n+1','n+2','n+3','n+4','n+5']
        with open('NEWRESULT/H0Resultoutput.csv', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            #write.writerow(fields)
            write.writerows(PercLists)
allKeys = tcas_names[col]
width= 0.15

d = []
with open('NEWRESULT/H0Resultoutput.csv') as csvfile:
    data = csv.reader(csvfile)
    max_elems = 0
    for row in data:
        if max_elems < len(row): max_elems = len(row)
    csvfile.seek(0)
    for i, row in enumerate(data):
        # fix my csv by padding the rows
        d.append(row + ["" for x in range(max_elems-len(row))])
#print(d)
df1 = pd.DataFrame(d, columns=['n+1','n+2','n+3','n+4','n+5','n+6','n+7','n+8'], dtype=np.dtype("float64"))
#df1 =df1.astype(np.float)
df1['n+1'] = pd.to_numeric(df1['n+1'], errors='coerce')
df1['n+2'] = pd.to_numeric(df1['n+2'], errors='coerce')
df1['n+3'] = pd.to_numeric(df1['n+3'], errors='coerce')
df1['n+4'] = pd.to_numeric(df1['n+4'], errors='coerce')
df1['n+5'] = pd.to_numeric(df1['n+5'], errors='coerce')
df1['n+6'] = pd.to_numeric(df1['n+6'], errors='coerce')
df1['n+7'] = pd.to_numeric(df1['n+7'], errors='coerce')
df1['n+8'] = pd.to_numeric(df1['n+8'], errors='coerce')
#['n+9'] = pd.to_numeric(df1['n+9'], errors='coerce')
df1 = df1.replace(np.nan, 0, regex=True)

#print(df1)

fig, ax = plt.subplots()
ax.bar(x - width, df1['n+1'], width, label='n+1', align='center')
ax.bar(x , df1['n+2'], width, label='n+2', align='center')
ax.bar(x + width, df1['n+3'], width, label='n+3',align='center')
ax.bar( x + 2 * width, df1['n+4'], width, label='n+4', align='center')
ax.bar( x + 3 * width, df1['n+5'], width, label='n+5', align='center')
ax.bar(x + 4 * width, df1['n+6'], width, label='n+6', align='center')
ax.bar(x + 5 * width, df1['n+7'], width, label='n+7', align='center')
ax.bar(x + 6 * width,  df1['n+8'], width, label='n+8', align='center')
#ax.bar(x + 7 * width,  df1['n+9'], width, label='n+9', align='center')

#plt.ylim([0,100])
ax.set_title('% of finding n+m test cases, H0, 1000 permutations')
ax.set_ylabel('Percentage of n+m TCs generated (%)')
ax.set_xlabel('Number conditions in a decision')
ax.set_xticks(x)
ax.set_xticklabels(labels.keys(), rotation = 'vertical', fontsize=8)
ax.legend()
plt.savefig('NEWRESULT/H0New1000.png', dpi=150)
plt.show()
#,'n+8','n+9'

