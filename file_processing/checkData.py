''' This .py file is responsible for checking the correctness of the data, considering the conditions and images'''

import os
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use([hep.style.CMS])


################################################################################################################################

data_path = '/net/scratch_cms3a/kann/fast_calo_flow/data/shielding_distance/'
result_path = '/home/home1/institut_3a/kann/Desktop/fast-simulations/calo_sim/GEANT4/shield_distance_test/'

# what part of the dataset to check
mode = 'validation'

################################################################################################################################

def check_uniform_distribution(arr, condition, bins=50, range_min=20_000, range_max=100_000):
    
    ''' Check if the values of a numpy array are uniformly distributed between range_min and range_max. '''

    print("-"*80)
    print(f"{condition} - HISTOGRAM CHECK: ")

    # Ensure the array values are within the range
    within_range = (arr >= range_min) & (arr <= range_max)
    if not np.all(within_range):
        print(f"Some values are outside the range [{range_min}, {range_max}].")

    fig = plt.figure()

    # Plot the histogram
    plt.hist(arr, bins=bins, range=(range_min, range_max), density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Histogram of Values (Range: {range_min} to {range_max})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.axhline(1 / (range_max - range_min), color='red', linestyle='--', label='Expected Uniform Density')
    plt.legend()
    fig.savefig(result_path + 'distribution_' + condition + '.pdf')

    # Compare histogram counts to uniformity
    counts, bin_edges = np.histogram(arr, bins=bins, range=(range_min, range_max), density=True)
    uniform_density = 1 / (range_max - range_min)  # Expected density for uniform distribution
    deviations = np.abs(counts - uniform_density)
    max_deviation = np.max(deviations)
    
    if max_deviation < 0.1 * uniform_density:  # Allow 10% deviation
        print("The array appears to be uniformly distributed.")
    else:
        print("The array does not appear to be uniformly distributed.")



images      = np.load(data_path + mode + '_img_'  + ".npy")
conditions  = np.load(data_path + mode + '_conditions_'  + ".npy")
labels      = np.load(data_path + mode + '_labels_' + ".npy")

# Printing shapes
print("Images shape:        ", images.shape)
print("Conditions shape:    ", images.shape)
print("Labels shape:        ", images.shape)


# Check Minimum and Maximum of Images
print("\nMinimum of images: ", np.min(images))
print("Maximum of images: ", np.max(images))

# Check if deposited energies are less than condition (initial particle energy)
max_values = np.max(images, axis = (1,2))
energy_condition = conditions[:,0]
print(np.sum((max_values > energy_condition)))
print(max_values[(max_values > energy_condition)])
print(energy_condition[(max_values > energy_condition)])

# Check label values:
print("-"*80)
print(f"LABEL CHECK: ")
unique_values, counts = np.unique(labels, return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"Value {value} occurs {count} times")


# Check NaN values:
number_nan_img          = np.isnan(images).sum()
number_nan_conditions   = np.isnan(conditions).sum()
number_nan_labels       = np.isnan(labels).sum()

print("Number of NaN values in images:      ", number_nan_img)
print("Number of NaN values in conditions:  ", number_nan_conditions)
print("Number of NaN values in labels:      ", number_nan_labels)


# Check the values of the conditions by plotting a histogram of the distribution

check_uniform_distribution(arr = conditions[:,0], condition = 'ENERGY', bins = 50, range_min=20_000, range_max=100_000)
check_uniform_distribution(arr = conditions[:,1], condition = 'THICKNESS', bins = 50, range_min=0.5, range_max=1.5)
check_uniform_distribution(arr = conditions[:,2], condition = 'DISTANCE', bins = 50, range_min=50, range_max=90)

# Plot examples
random_int = np.random.randint(low = 0, high = images.shape[0] - 15)

for i in range(random_int, random_int + 15):

    fig, ax = plt.subplots(figsize = (12,10))

    neg_sample = np.ma.masked_greater_equal(images[i], 0) 
    pos_sample = np.ma.masked_less_equal(images[i], 0)

    cmap = copy.copy(matplotlib.colormaps["viridis"])

    # Use LogNorm to apply logarithmic scaling to color mapping
    norm = mcolors.LogNorm(vmin=1, vmax=100e3)


    # Negative Werte in Rot anzeigen
    neg_cmap = mcolors.LinearSegmentedColormap.from_list("neg_cmap", ["red", "white"])
    neg_plot = ax.imshow(neg_sample, cmap=neg_cmap, vmin=-1000, vmax=0)

    # Positive Werte mit viridis und logarithmischer Skala anzeigen
    pos_plot = ax.imshow(pos_sample, cmap=cmap, norm=norm)

    # Remove ticks and tick labels from the main image plot
    ax.set_xticks([])
    ax.set_yticks([])

    # Farbbalken rechts und links vom Plot hinzufügen
    divider = make_axes_locatable(ax)

    # Rechter Farbbalken für positive Werte
    cax_pos = divider.append_axes("right", size="5%", pad=0.15)
    cbar_pos = fig.colorbar(pos_plot, cax=cax_pos)
    cbar_pos.set_label("Energy [MeV]")

    # Linker Farbbalken für negative Werte
    cax_neg = divider.append_axes("left", size="5%", pad=0.15)
    cbar_neg = fig.colorbar(neg_plot, cax=cax_neg)
    #cbar_neg.set_ticks([neg_sample.min(), 0])

    cbar_neg.ax.yaxis.set_label_position('left')
    cbar_neg.ax.yaxis.tick_left()

    ax.set_title(f"Energy: {conditions[i,0]:.1f}, Thickness: {conditions[i,1]:.2f}, Distance: {conditions[i,2]:.2f}")
    plt.imshow(images[i], cmap = cmap, norm=norm)

    plt.savefig(result_path + f"test_image{i}.pdf")