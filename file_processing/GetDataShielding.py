import os
import copy

import numpy as np
import pandas as pd
import numba

from sklearn.model_selection import train_test_split


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.CMS])



############################ Helper functions ###################################


def save_energy_grid_plot(energy_grid, filepath):
    
    ''' Function for plotting a shower shape image.'''

    # colormaps modification
    cmap = copy.copy(matplotlib.colormaps["viridis"])
    cmap.set_under('w')


    # Use LogNorm to apply logarithmic scaling to color mapping
    norm = mcolors.LogNorm(vmin=1, vmax=100e3)


    plt.imshow(energy_grid, norm = norm, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Energy [MeV]")
    plt.title("Energy Grid")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the plot to the specified filepath
    plt.savefig(filepath)
    plt.close()  # Close the plot to free up memory



@numba.jit
def find_shower_numba(LR, LR_dim, window_size_x, window_size_y):

    """
    Finds the index of the corner of the window with the lowest energy resolution
    """


    shower_energy = np.zeros((LR_dim-window_size_x, LR_dim-window_size_y)) # understand indices as the upper left corner of the window_size*window_size window that will be selected from the image, value will be the sum of the window energy

    for x in range(LR_dim-window_size_x):

        for y in range(LR_dim-window_size_y):

            shower_energy_xy = LR[x:x+window_size_x, y:y+window_size_y].sum()

            shower_energy[x,y] = shower_energy_xy

    return np.argmax(shower_energy)


def find_shower(image_LR, LR_dim = 24, window_size_x=12, window_size_y = 12):

    """
    Cuts out the window_size x window_size image with the highest energy,
    this will be the LR photon or pion object.
    """


    shower_corner = find_shower_numba(image_LR, LR_dim, window_size_x, window_size_y)

    x_window, y_window = np.unravel_index(shower_corner, (LR_dim-window_size_x, LR_dim-window_size_y)) #corner of best window

    LR_cut = image_LR[x_window:x_window+window_size_x, y_window:y_window+window_size_y]

    # compute the deviation of energy in percent

    original_energy = np.sum(image_LR)
    cut_energy = np.sum(LR_cut)

    error = cut_energy/original_energy

    return LR_cut, error

def cut_images(shower_images, original_size, window_size_x, window_size_y):

    ''' Cut out the images for the calculated windows. '''

    cut_images = np.zeros(shape=(shower_images.shape[0],window_size_x,window_size_y))
    relative_error = np.zeros(shape=(shower_images.shape[0],))

    # Iterate over images

    for idx, shower in enumerate(shower_images):

        cut_image, relative_error_value = find_shower(shower, original_size, window_size_x=window_size_x, window_size_y=window_size_y)
        cut_images[idx] = cut_image
        relative_error[idx] = relative_error_value

    # For an estimation how much energy we loose

    sorted_relative_error = np.sort(relative_error)

    print("Mean Relative Error:             ", np.mean(relative_error))
    print("Median Relative Error:           ", np.median(relative_error))
    print("Highest Relative Error:          ", sorted_relative_error[0])
    print("5. Highest Relative Error:       ", sorted_relative_error[4])
    print("100. Highest Relative Error:     ", sorted_relative_error[99])
    print("1000. Highest Relative Error:    ", sorted_relative_error[999])
    print("5000. Highest Relative Error:    ", sorted_relative_error[4999])
    print("10000. Highest Relative Error:   ", sorted_relative_error[9999])
    print("20000. Highest Relative Error:   ", sorted_relative_error[19999])

    return cut_images


def convert_energy_grid_numpy(dataframe, column_name):

    ''' Convert the energy grids to numpy arrays. '''

    # convert to 24 x 24 numpy arrays
    dataframe[column_name + '_array'] = dataframe[column_name].apply(lambda x: np.stack(x))
    dataframe.drop(columns = [column_name], inplace = True)

    return dataframe


def get_labels(dataframe, column_name):

    ''' Take the dataframe and return an array with labels for the particle types. '''

    particle_types = np.array(dataframe[column_name].values, dtype='<U3')

    binary_labels = np.where(np.char.startswith(particle_types, 'pion'), 1, 0)

    return binary_labels


def get_energy_conditions_as_sum(images):

    ''' Get energy condition as sum of image. '''

    energy_condition = np.sum(images, axis = (1,2))

    return energy_condition


def get_condition_from_dataframe(dataframe, column_name):
    
    ''' Take the dataframe and return an array with condition. '''

    condition = np.array(dataframe[column_name].values)

    return condition


def expand_dataframe(dataframe, column_name):

    ''' Get the particle properties as condition from the dictionary '''

    # Since each entry in 'four_vector' is a list of 1 dictionary, we apply a lambda function to extract the dictionary
    expanded_df = pd.json_normalize(dataframe[column_name].apply(lambda x: x[0]))

    # Now we can concatenate the expanded columns with the original dataframe (dropping 'four_vector' if needed)
    df_expanded = pd.concat([dataframe.drop(columns=[column_name]), expanded_df], axis=1)

    return df_expanded



def merge_conditions(*args):

    ''' Merge all different conditions (energy, noise, shielding) together. '''

    # Validate input shapes
    for arr in args:
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"All inputs must have shape (n, 1). Got {arr.shape} instead.")

    # Stack arrays horizontally
    return np.hstack(args)


def train_test_val_split(images, labels, conditions, ratio = (0.55, 0.4, 0.05)):

    ''' Perform train test validation splitting. '''

    random_state = 42

    train_data, test_val_data, train_labels, test_val_labels, train_conditions, test_val_conditions = train_test_split(images, labels, conditions, shuffle=True, random_state=random_state, train_size=ratio[0])

    test_data, val_data, test_labels, val_labels, test_conditions, val_conditions = train_test_split(test_val_data, test_val_labels, test_val_conditions, shuffle=True, random_state=random_state, train_size=ratio[1]/(ratio[1] + ratio[2]))

    return train_data, test_data, val_data, train_labels, test_labels, val_labels, train_conditions, test_conditions, val_conditions


def add_noise(images, labels, conditions, image_size_x, image_size_y, min_std = 10, max_std = 200, double_size = True):

    ''' This function adds noise to the images. '''

    mean = 0

    rng = np.random.default_rng(seed = 42)

    if double_size:
        noise_range1, noise_range2 = (min_std, min_std + (max_std - min_std)/2), (min_std + (max_std - min_std)/2, max_std)

        std1 = rng.uniform(*noise_range1, size = (images.shape[0],))
        std2 = rng.uniform(*noise_range2, size = (images.shape[0],))

        noise1 = rng.normal(loc = mean, scale = std1[:, None, None], size = (images.shape[0], image_size_x, image_size_y))
        noise2 = rng.normal(loc = mean, scale = std2[:, None, None], size = (images.shape[0], image_size_x, image_size_y))

        images1 = images + noise1
        images2 = images + noise2

        images = np.concatenate((images1, images2))
        labels = np.concatenate((labels, labels))
        conditions = np.concatenate((conditions, conditions))

        noise_condition = np.concatenate((std1, std2), axis = 0).reshape(conditions.shape[0],1)

        

    else:
        
        noise_range = (min_std, max_std)

        std = rng.uniform(*noise_range, size = (images.shape[0],))

        noise = rng.normal(loc = mean, scale = std[:, None, None, None], size = (images.shape[0], image_size_x, image_size_y, 1))

        images = images + noise

        noise_condition = std.reshape(conditions.shape[0],1)

    # add noise to condition
    conditions = np.concatenate((conditions, noise_condition), axis = 1)


    return images, labels, conditions

def seperate_particles(images, labels, conditions, particle):

    if particle:

        boolean = ['photon', 'pion'].index(particle)

        mask = (labels == boolean)

        images      = images[mask]
        labels      = labels[mask]
        conditions  = conditions[mask]

        return images, labels, conditions
    
    else:

        return images, labels, conditions

def filter_out_energies(images, labels, conditions, min_energy):

    energies = conditions[:,0]

    mask = (energies > min_energy)

    images = images[mask]
    labels = labels[mask]
    conditions = conditions[mask]

    return images, labels, conditions



def save_as_arrays(data_path, train_images, train_labels, train_conditions, test_images, test_labels, test_conditions, validation_images, validation_labels, validation_conditions, noise = False, particle = None):

    ''' Function for saving the arrays in the given datapath'''

    if noise: data_path = data_path + "noise/"
    os.makedirs(data_path, exist_ok=True)

    if not particle: particle = ''

    np.save(data_path + 'train_img_' + particle + ".npy", train_images)
    np.save(data_path + 'train_labels_' + particle + ".npy", train_labels)
    np.save(data_path + 'train_conditions_' + particle + ".npy", train_conditions)

    np.save(data_path + 'test_img_' + particle + ".npy", test_images)
    np.save(data_path + 'test_labels_' + particle + ".npy", test_labels)
    np.save(data_path + 'test_conditions_' + particle + ".npy", test_conditions)

    np.save(data_path + 'validation_img_' + particle + ".npy", validation_images)
    np.save(data_path + 'validation_labels_' + particle + ".npy", validation_labels)
    np.save(data_path + 'validation_conditions_' + particle + ".npy", validation_conditions)






############################ Main program ###################################

window_size_x = 24
window_size_y = 24
original_size = 24
noise = False           # right now irrelevant
augment_data = True     # only if noise is true
specify_particle = 'photon'

# Directory to store the .npy files
data_path = "/net/data_cms3a-1/kann/fast_calo_flow/data/shielding_distance_24x24/"

# Directory where Parquet files are stored
parquet_dir = "/net/data_cms3a-1/kann/fast_calo_flow/raw_files/shielding_distance/"



def main():

    # read in file(s)
    dataframe = pd.read_parquet(parquet_dir)

    # transform 'four-vector' dict to columns
    dataframe = expand_dataframe(dataframe=dataframe, column_name='four_vector')


    # convert energy grids to numpy
    dataframe = convert_energy_grid_numpy(dataframe=dataframe, column_name='energy_grid')

    # get the binary labels from the particle type
    labels = get_labels(dataframe=dataframe, column_name='name')

    # get all the shower images in an array
    shower_images = np.stack(dataframe['energy_grid_array'].values)

    # cut out the relevant part of the shower
    # images = cut_images(shower_images=shower_images, original_size=original_size, window_size_x=window_size_x, window_size_y=window_size_y)
    images = shower_images

    ### get conditions
    condition_energy = get_condition_from_dataframe(dataframe=dataframe, column_name='E').reshape(-1,1)
    condition_thickness = get_condition_from_dataframe(dataframe=dataframe, column_name='thickness').reshape(-1,1)
    condition_distance  = get_condition_from_dataframe(dataframe=dataframe, column_name = 'distance_to_detector').reshape(-1,1)


    # transform to radiation lengths:
    condition_thickness = condition_thickness / 1.757

    conditions = merge_conditions(condition_energy, condition_thickness, condition_distance)

    # images, labels, conditions = filter_out_energies(images=images, labels=labels, conditions=conditions, min_energy=10_000)

    
    # get separate files for different particles
    images, labels, conditions = seperate_particles(images, labels, conditions, particle=specify_particle)


    if noise: images, labels, conditions = add_noise(images=images, labels=labels, conditions=conditions, image_size_x=window_size_x, image_size_y=window_size_y, min_std=10, max_std=200, double_size=augment_data)

    train_images, test_images, validation_images, train_labels, test_labels, validation_labels, train_conditions, test_conditions, validation_conditions = train_test_val_split(images = images, labels = labels, conditions = conditions, ratio = (0.60,0.30,0.1))

    save_as_arrays(data_path=data_path, train_images=train_images, train_labels= train_labels, train_conditions=train_conditions, test_images=test_images, test_labels=test_labels, test_conditions=test_conditions, validation_images=validation_images, validation_labels=validation_labels, validation_conditions=validation_conditions, noise=noise, particle=specify_particle)



if __name__ == '__main__':

    main()
