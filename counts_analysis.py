import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv

data_file1 = 'counts_Data/dos_led_back.txt'
data_file2 = 'counts_Data/dos_osl.txt'

salt_file1 = 'counts_Data/'

def parse_txt(data_file):
    times = []
    counts = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            times.append(float(row[0]))
            counts.append(float(row[1]))
    return [times, counts]

data1 = parse_txt(data_file1)
times1 = list((np.array(data1[0]) * 1e-3) - data1[0][0] * 1e-3)
counts1 = list(np.array(data1[1]) - data1[1][0])

data2 = parse_txt(data_file2)
times2 = list((np.array(data2[0]) * 1e-3) - data2[0][0] * 1e-3)
counts2 = list(np.array(data2[1]) - data2[1][0])

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 18,
        }

plt.plot(times1, counts1, label='Background (with LED)')
plt.plot(times2, counts2, label='OSL')
plt.legend(prop={'family': 'serif'}, edgecolor = "black", fancybox=False)
plt.xlabel('Time (s)', fontdict=font)
plt.ylabel('Cumulative Counts', fontdict=font)



# plt.savefig(fname = my_file_name)
plt.show()