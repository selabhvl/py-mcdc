import ast
import csv
from itertools import repeat
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

from tcasii import tcas


def plot(allKeys, chart_name, resultMap, t):
    # We might want to compute `allKeys` in here instead of outside...
    print('Generating GNUplot in \'{}.csv\'. Use `gnuplot -p \'{}.plot\'` to print.\n'
          .format(chart_name, chart_name), file=sys.stderr)
    csvMap = {}
    # Initialise all keys:
    for k in allKeys:
        csvMap[k] = []
    # now print the bloody thing:
    with open('{}.csv'.format(chart_name), 'w') as file:
        with open('{}.plot'.format(chart_name), 'w') as plotfile:
            plotfile.write('set datafile separator \',\'\nset key autotitle columnhead\nplot \\\n')
            file.write('|tc|,')
            for col, (f, rm) in enumerate(resultMap.items()):
                thisRounds = sum(dict(rm).values()) # Haha, do you even Python?!
                plotfile.write('\'{}.csv\' using 1:{} with lines,\\\n'.format(chart_name, col + 2))
                # print labels for autotitle, +1 since we start with D1
                file.write('{}:{},'.format(col + 1, len(f.inputs) + 1))
                for k in allKeys:
                    try:
                        # We had "normalized" above, so now we need to "rehydrate":
                        v = dict(rm)[k + len(f.inputs) + 1] / thisRounds  # Ick
                    except:
                        v = 0
                    csvMap[k].append(v)
            file.write('\n')  # finished printing labels.
            for k in csvMap:
                v = csvMap[k]
                file.write('{},'.format(k))
                for c in v:
                    file.write('{},'.format(c))
                file.write('\n')

    # Matplotlib goes here:
    plotting = True
    if plotting:
        # Create new figure:
        fig = plt.figure(chart_name, clear=True)
        ax = fig.subplots()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        # Set limits (min, max) in the axes
        #ax.set_title('% of finding n+m test cases, H1, 1000 permutations')
        ax.set_ylabel('Probability distribution')
        ax.set_xlabel('Number of tests (m) additional to minimum set n+1')
        ax.set_ylim(0.0, 1.0)
        for i, (k_f, v_rm) in enumerate(resultMap.items()):
            expected_value = sum(map(lambda kv: (kv[0] - len(k_f.inputs)) * kv[1], v_rm))
            if v_rm[0][0] == len(k_f.inputs)+1:
                label_i = str(i)+':'+str(v_rm[0][1])  # Print only no. of n+1s in legend
                # label_i = str(i) + ':' + str(expected_value)
            else:
                label_i = str(i)+':'+str(expected_value)+'*'
            ax.plot(list(csvMap.keys()), list(map(lambda v: v[i], csvMap.values())), label=label_i)
        if t is not None:
            ax.text(0, 1.1, r'total time: {0}'.format(t), size=12, math_fontfamily='dejavuserif', horizontalalignment='left',
                    verticalalignment='center')

        fig.legend()
        plt.draw()


def load_resultMap(csvname):
    resultMap = dict()
    allKeys = set()

    with open(csvname, newline='') as csvfile:
        result_map_reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result_map_reader:
            # col = 1
            # rm = [(8, 4), (9, 10), (10, 1)]
            col, rm = int(row[0]), ast.literal_eval(row[1])
            f = tcas[col]
            resultMap[f] = rm

            # Populate allKeys
            # TODO: Maybe we shouldn't have the data in this way...
            new_keys = set(i - len(f.inputs) - 1 for (i, _) in resultMap[f])
            allKeys.update(new_keys)

        return resultMap, allKeys


if __name__ == "__main__":
    csvname = sys.argv[1]
    chartName = csvname.replace('_resultMap.csv', '')
    resultMap, allKeys = load_resultMap(csvname)
    # Plot with no timestamp data:
    plot(allKeys, chartName, resultMap, repeat(None))
