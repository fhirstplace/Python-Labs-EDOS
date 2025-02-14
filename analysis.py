import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

data_file = 'Data/mosfet_beta.xml'
tree = ET.parse(data_file) 
root = tree.getroot()
datasets = []

for dataset in root:
    currents = []
    current_std = []
    voltages = []
    voltage_std = []
    for entry in dataset:
        if entry.tag == 'Current':
            currents.append(float(entry[0].text))
            current_std.append(float(entry[1].text))
        elif entry.tag == 'Voltage':
            voltages.append(float(entry[0].text))
            voltage_std.append(float(entry[1].text))
    item = [[currents, current_std], [voltages, voltage_std]]
    datasets.append(item)


times = range(0,len(root))
# z = np.polyfit(voltages, currents, 1)
# Y1p = np.poly1d(z)

threshold_voltages = []
for dataset in datasets:
    currents = dataset[0][0]
    current_std = dataset[0][1]
    voltages = dataset[1][0]
    voltage_std = dataset[1][1]
    largest_change_in_current = 0
    threshold_voltage = 0
    for i in range(0, len(currents) - 1):
        change_in_current = currents[i+1] - currents[i] 
        if change_in_current > largest_change_in_current:
            largest_change_in_current = change_in_current
            threshold_voltage = (voltages[i] + voltages[i+1]) / 2
    threshold_voltages.append(threshold_voltage)

plt.errorbar(times, threshold_voltages)
title = "Vgs vs time"
# plt.legend()
plt.title(title)
plt.ylabel('Vgs')
plt.xlabel('t (hrs)')
plt.savefig(fname = "Graphs/"+data_file[5:-4]+"VgsT.png")
plt.show()
