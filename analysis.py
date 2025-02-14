import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

data_file = 'Data/uniradiated_mosfetVDS.xml'
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

i = 0
for dataset in datasets:
    currents = dataset[0][0]
    current_std = dataset[0][1]
    voltages = dataset[1][0]
    voltage_std = dataset[1][1]
    print(i)
    plt.errorbar(voltages,currents,xerr = voltage_std, yerr = current_std, color = (i/70, 0, 1 - i/70))
    i+=1

title = "Id vs VDS for a MOSFET with constant Vgs=3.7V"
# plt.legend()
plt.title(title)
plt.ylabel('Ids')
plt.xlabel('Vds')
plt.savefig(fname = "Graphs/"+data_file[5:-4]+".png")
plt.show()
