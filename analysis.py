import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

data_file = 'Data/mosfet_beta92.xml'
tree = ET.parse(data_file) 
root = tree.getroot()
datasets = []

#gathering data from xml file
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

#analysing data
threshold_voltages = []
threshold_voltages_errors = []
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
            threshold_voltages_error = np.abs((voltages[i+1] - voltages[i])/2)
    threshold_voltages.append(threshold_voltage)
    threshold_voltages_errors.append(threshold_voltages_error)

#times in hrs
#times = range(0,len(root))
times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 91]

#modelling
z = np.polyfit(times, threshold_voltages, 1)
Y1p = np.poly1d(z)

#plotting
plt.errorbar(times, threshold_voltages, yerr = threshold_voltages_errors, label = "data")
plt.errorbar(times, Y1p(times), fmt = '.', color = "black", label = 'model')
title = "MOSFET Vgs vs time irradiated"
plt.legend()
plt.title(title)
plt.ylabel('Vgs')
plt.xlabel('t (hrs)')
plt.savefig(fname = "Graphs/"+data_file[5:-4]+"VgsT.png")
plt.show()
