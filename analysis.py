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

# make graph things
title = "Id vs Vg for a MOSFET"
plt.legend()
plt.title(title)
plt.ylabel('Id')
plt.xlabel('Vg')
plt.savefig(fname = "Graphs/"+data_file[5:-4]+".png")
plt.show()
