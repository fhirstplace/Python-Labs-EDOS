import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

iv_graph = False
steepest_line = False
derivative = False
dvt_over_time = False
lvt_over_time = True
linear_fit = False

if linear_fit:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [4, 1]})
    fig.subplots_adjust(hspace=0)
else:
    fig, ax = plt.subplots()
    ax = [ax]

if (iv_graph or steepest_line or derivative) and (dvt_over_time or lvt_over_time):
    print("Please only select compatible graph types")
    exit()

#parse xml file
data_file = 'Data/ba133_iradiated_linear.xml'
data_file2 = None

#parse data
def xml_to_datasets(data_file):
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
    return datasets

#plot data
def plot_iv(datasets):
    i = 0
    for dataset in datasets:
        currents = dataset[0][0]
        current_std = dataset[0][1]
        voltages = dataset[1][0]
        voltage_std = dataset[1][1]
        if i % 10 == 0:
            ax[0].errorbar(voltages,currents,xerr = voltage_std, yerr = current_std, label = str("t="+str(i)), color = (i/len(datasets), 0, 1 - i/len(datasets)))
        else:
            pass
            #ax[0].errorbar(voltages,currents,xerr = voltage_std, yerr = current_std, color = (i/len(datasets), 0, 1 - i/len(datasets)))
        i+=1

#linear fit at steepest point to find threshold voltage
def plot_steepest_line(datasets):
    for dataset in datasets:
        currents = dataset[0][0]
        current_std = dataset[0][1]
        voltages = dataset[1][0]
        voltage_std = dataset[1][1]
        max_derivative = 0
        max_derivative_index = 0
        for i in range(len(currents) - 1):
            if voltages[i + 1] - voltages[i] > 0.01:
                gradient = (currents[i + 1] - currents[i]) / (voltages[i + 1] - voltages[i])
                if gradient > max_derivative and currents[i] > 0:
                    max_derivative = gradient
                    max_derivative_index = i
                    gradient_err = gradient * np.sqrt((current_std[i] / currents[i])**2 + (voltage_std[i + 1] / voltages[i + 1])**2 + (voltage_std[i] / voltages[i])**2)
        f = lambda x : max_derivative * (x - voltages[max_derivative_index]) + currents[max_derivative_index]
        ax[0].errorbar(voltages, list(map(f,(voltages))),fmt="--", color = 'red')

        #error calculations
        x_intercept = (-currents[max_derivative_index] / max_derivative) + voltages[max_derivative_index]
        error = np.sqrt((currents[max_derivative_index] / max_derivative) * np.sqrt((current_std[max_derivative_index] / currents[max_derivative_index])**2 + (gradient_err / max_derivative)**2)**2 + voltage_std[max_derivative_index]**2)
        print("Threshold voltage: ", x_intercept, "+/-", error)

def calculate_derivative(x, y):
    derivative = np.array([])
    for i in range(1, len(y) - 1):
        if x[i + 1] - x[i - 1] > 0.02:
            gradient = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
            derivative = np.append(derivative, gradient)
    return list(derivative)

def calculate_smoothed_derivative(x, y):
    derivative = np.array([])
    r = 7
    NC = r * (r + 1) * (2 * r + 1) / 3
    for i in range(r, len(y) - r - 1):
        sum = 0
        for j in range(-r, r + 1):
            sum += y[i + j] * j
        derivative = np.append(derivative, (sum / NC) * (1/((x[i+1]-x[i])+(x[i]-x[i-1]))/2))
    return list(derivative)

#calculate and plot derivative
def plot_derivative(datasets):
    for dataset in datasets:
        currents = dataset[0][0]
        voltages = dataset[1][0]
        derivative = calculate_smoothed_derivative(voltages, currents)
        offset = int(-(len(derivative) - len(voltages)) / 2)
        voltages = voltages[offset:-offset]
        derivative2 = calculate_derivative(voltages, derivative)
        max_current = max(currents)
        scaled_derivative2 = np.array(derivative2) * (max_current / max(derivative2))
        offset = int(-(len(derivative2) - len(voltages)) / 2)
        print("offset:", offset)
        voltages = voltages[offset:-offset-1]
        print("voltages:", len(voltages), "derivative2:", len(derivative2))
        ax[0].errorbar(voltages,scaled_derivative2,fmt="--", color = 'green')
        index_max_derivative2 = derivative2.index(max(derivative2))
        max_derivative2_voltage = voltages[index_max_derivative2]
        error = max((voltages[index_max_derivative2 + 1]) - max_derivative2_voltage, max_derivative2_voltage - voltages[index_max_derivative2 - 1])
        print("Second Derivative Maximum: ", max_derivative2_voltage, "+/-", error)

#calculate and plot derivative calculation of threshold voltages over time
def plot_derivative_thresholds_time(datasets, times=None, colour = "green", label="2nd derivative method"):
    max_derivative2_voltages = []
    errors = []
    if times == None:
        times = range(0, len(datasets))
    for dataset in datasets:
        currents = dataset[0][0]
        current_std = dataset[0][1]
        voltages = dataset[1][0]
        voltage_std = dataset[1][1]
        derivative = calculate_smoothed_derivative(voltages, currents)
        offset = int(-(len(derivative) - len(voltages)) / 2)
        voltages = voltages[offset:-offset]
        derivative2 = calculate_derivative(voltages, derivative)
        offset = int(-(len(derivative2) - len(voltages)) / 2)
        voltages = voltages[offset:-offset-1]
        index_max_derivative2 = derivative2.index(max(derivative2))
        max_derivative2_voltages.append(voltages[index_max_derivative2])
        errors.append(max((voltages[index_max_derivative2 + 1]) - voltages[index_max_derivative2], voltages[index_max_derivative2] - voltages[index_max_derivative2 - 1]))
    ax[0].errorbar(times, max_derivative2_voltages, yerr = errors, color = colour, label = label)

def plot_steepest_threshold_time(datasets, times=None, colour = "red", label="steepest line method"):
    x_intercepts = []
    errors = []
    if times == None:
        times = range(0, len(datasets))
    for dataset in datasets:
        currents = dataset[0][0]
        current_std = dataset[0][1]
        voltages = dataset[1][0]
        voltage_std = dataset[1][1]
        max_derivative = 0
        max_derivative_index = 0
        for i in range(len(currents) - 1):
            if voltages[i + 1] - voltages[i] > 0.01:
                gradient = (currents[i + 1] - currents[i]) / (voltages[i + 1] - voltages[i])
                if gradient > max_derivative and currents[i] > 0:
                    max_derivative = gradient
                    max_derivative_index = i
                    gradient_err = gradient * np.sqrt((current_std[i] / currents[i])**2 + (voltage_std[i + 1] / voltages[i + 1])**2 + (voltage_std[i] / voltages[i])**2)
        f = lambda x : max_derivative * (x - voltages[max_derivative_index]) + currents[max_derivative_index]
        #error calculations
        x_intercept = (-currents[max_derivative_index] / max_derivative) + voltages[max_derivative_index]
        error = np.sqrt(0.005**2 + (currents[max_derivative_index] / max_derivative) * np.sqrt((current_std[max_derivative_index] / currents[max_derivative_index])**2 + (gradient_err / max_derivative)**2)**2 + voltage_std[max_derivative_index]**2)
        x_intercepts.append(x_intercept)
        errors.append(error)
    ax[0].errorbar(times, x_intercepts, yerr = errors, color = colour, label=label)
    return [x_intercepts, errors]

datasets = xml_to_datasets(data_file)
if data_file2 != None:
    datasets2 = xml_to_datasets(data_file2)

# make graph things
if iv_graph:
    plot_iv(datasets)
    title = "Id vs Vg for a MOSFET"
    ax[0].ylabel('Id')
    ax[0].xlabel('Vg')
if steepest_line:
    plot_steepest_line(datasets)
    ax[0].xlim(1.35, 2.85)
    ax[0].ylim(0, 0.00015)
if derivative:
    plot_derivative(datasets)
if dvt_over_time:
    plot_derivative_thresholds_time(datasets)
    if data_file2 != None:
        plot_derivative_thresholds_time(datasets2, times=range(len(datasets), len(datasets) + len(datasets2)),colour = "blue",label="2nd derivative method (no source)")
    title = "Threshold Voltage vs Time"
    ax[0].xlim(left=0)
    ax[0].ylabel('Vt')
    ax[0].xlabel('Time hrs')
if lvt_over_time:
    title = "Threshold Voltage vs Time"
    threshold_voltages = plot_steepest_threshold_time(datasets)
    times = range(0,len(datasets))
    if data_file2 != None:
        threshold_voltages.append(plot_steepest_threshold_time(datasets2, times=range(len(datasets), len(datasets) + len(datasets2)),colour = "Purple", label="steepest line method (no source)"))[0]
        times = range(0,len(datasets) + len(datasets2))
    if linear_fit:
        z = np.polyfit(times, threshold_voltages[0], 1)
        Y1p = np.poly1d(z)
        ax[0].errorbar(times, Y1p(times), fmt = '--', color = "black", label = 'model')
        chi_2 = 0
        for i in range(len(threshold_voltages[0])):
            chi_2 += (threshold_voltages[0][i] - Y1p(times[i]))**2 /(threshold_voltages[1][i]**2)
        #add residuals
        Res_1s = np.array(threshold_voltages[0]) - np.array(Y1p(times))

        #plot residuals
        ax[1].plot(times, Res_1s/np.std(Res_1s), marker = '.', color = "red",linewidth = 0)
        ax[1].plot(times, np.zeros(len(times)), color = "black", linestyle = "--")

        print("chi^2:", chi_2)
        print("Normalised chi^2:", chi_2 / (len(threshold_voltages[0]) - 2))
        print("The prob of this occuring by chance is:", 1 - stats.chi2.cdf(chi_2, len(threshold_voltages[0]) - 2))

#define file name
if iv_graph:
    file_name = "Graphs/" + data_file[5:-4] + "_iv" + ".png"
if dvt_over_time or lvt_over_time:
    file_name = "Graphs/" + data_file[5:-4] + "_VtT" + ".png"
if data_file2 != None:
    file_name = "Graphs/" + data_file[5:-4] + "_VtT" + "_combined" + data_file[5:-4] + ".png"
#making graph details
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 18,
        }
if linear_fit:
    ax[1].set_ylabel("Normalised \nResiduals", fontdict = font)
    ax[1].set_xlabel('Time (hrs)', fontdict = font)
else:
    ax[0].set_xlabel('Time (hrs)', fontdict = font)
ax[0].set_ylabel("Threshold Voltage (V)", fontdict = font)
ax[0].legend(prop={'family': 'serif'}, edgecolor = "black", fancybox=False)
ax[0].set_title(title, fontdict = font)
plt.savefig(fname = file_name)
plt.show()