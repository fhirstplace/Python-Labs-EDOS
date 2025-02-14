import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

tree = ET.parse('Data/mosfet_beta.xml') 
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
z = np.polyfit(voltages, currents, 1)
Y1p = np.poly1d(z)

i = 0
for dataset in datasets:
    currents = dataset[0][0]
    current_std = dataset[0][1]
    voltages = dataset[1][0]
    voltage_std = dataset[1][1]
    plt.errorbar(voltages,currents,xerr = voltage_std, yerr = current_std, label=str("t = "+str(i)+" hrs"), color = (i/70, 0, 1 - i/70))
    i+=1

plt.legend()
plt.title("Id vs Vgs for a MOSFET irradiated over 68 hours with beta particles")
plt.ylabel('Id')
plt.xlabel('Vgs')
plt.show()

