import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#parse xml file
data_file = 'Data/Ba133_uniradiated_linear.xml'
tree = ET.parse(data_file) 
root = tree.getroot()
datasets = []

#parse data
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

#plot data
i = 0
for dataset in datasets:
    currents = dataset[0][0]
    current_std = dataset[0][1]
    voltages = dataset[1][0]
    voltage_std = dataset[1][1]
    if i % 10 == 0:
        plt.errorbar(voltages,currents,xerr = voltage_std, yerr = current_std, label = "uniradiated", color = (i/71, 0, 1 - i/71))
    else:
        plt.errorbar(voltages,currents,xerr = voltage_std, yerr = current_std, color = (i/71, 0, 1 - i/71))
    i+=1

#linear fit at steepest point to find threshold voltage
for dataset in datasets:
    currents = dataset[0][0]
    current_std = dataset[0][1]
    voltages = dataset[1][0]
    voltage_std = dataset[1][1]
    max_derivative = 0
    max_derivative_index = 0
    for i in range(len(currents) - 1):
        gradient = (currents[i + 1] - currents[i]) / (voltages[i + 1] - voltages[i])
        if gradient > max_derivative:
            max_derivative = gradient
            max_derivative_index = i
    f = lambda x : max_derivative * (x - voltages[max_derivative_index]) + currents[max_derivative_index]
    plt.errorbar(voltages, list(map(f,(voltages))),fmt="--", color = 'red')
    print("x-intercept = " + str((-currents[max_derivative_index] / max_derivative) + voltages[max_derivative_index]))


# make graph things
title = "Id vs Vg for a MOSFET"
plt.legend()
plt.title(title)
plt.ylabel('Id')
plt.xlabel('Vg')
plt.xlim(1.35, 2.85)
plt.ylim(0, 0.00015)
plt.savefig(fname = "Graphs/"+data_file[5:-4]+".png")
plt.show()
