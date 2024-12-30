import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.CMS])

import numpy as np

data_path = "/net/scratch_cms3a/kann/fast_calo_flow/data_florian/"




size = 60_000
noise_levels = np.array([10,20,50,75,100,150,200])
colors = ['purple', 'navy', 'mediumseagreen', 'darkorange', 'red', 'gold', 'maroon']
def downsample(samples):
    reshaped_images = samples.reshape(size, 12, 2, 12, 2)
        
    # Sum over the second and fourth dimensions (neighboring pixels)
    downsized_images = reshaped_images.sum(axis=(2, 4))

    return downsized_images

def get_binedges(bin_centers):

    bin_edges = np.concatenate((
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[1:] + bin_centers[:-1]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]
    ))

    return bin_edges

def get_barycenter(images):

    total_energy    = np.sum(images, axis=(1,2))

    x_coords, y_coords = np.meshgrid(np.arange(images.shape[2]), np.arange(images.shape[1]))

    # Expand dimensions to match sample size
    x_coords = np.broadcast_to(x_coords, images.shape)
    y_coords = np.broadcast_to(y_coords, images.shape)

    barycenter_x    =  np.sum(x_coords * images, axis=(1, 2)) / total_energy
    barycenter_y    =  np.sum(y_coords * images, axis=(1, 2)) / total_energy

    return barycenter_x, barycenter_y

def shower_width(images):

    total_energy    = np.sum(images, axis=(1,2))

    barycenter_x, barycenter_y = get_barycenter(images)


    x_coords, y_coords = np.meshgrid(np.arange(images.shape[2]), np.arange(images.shape[1]))

    # Expand dimensions to match sample size
    x_coords = np.broadcast_to(x_coords, images.shape)
    y_coords = np.broadcast_to(y_coords, images.shape)

    # Calculate differences to barycenter, expanded for sample size
    delta_x = x_coords - barycenter_x[:, None, None]
    delta_y = y_coords - barycenter_y[:, None, None]

    # calculate angular distances
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # get shower width
    numerator = np.sum(distance*images, axis = (1,2))
    shower_width = numerator / total_energy

    return shower_width


import numpy as np

def calculate_r9(images):
    """
    Calculate R9 values for a batch of 12x12 calorimeter images using vectorized operations.

    Parameters:
        images:              A 3D array of shape (n, 12, 12), 
                             where n is the number of images, 
                             and each 12x12 represents energy deposits.

    Returns:
        A 1D array of R9 values for each image.
    """

    
    # Total energy for each image
    E_total = images.sum(axis=(1, 2))  # Sum over rows and columns


    # Find the indices of the maximum energy cell for each image
    max_indices = np.argmax(images.reshape(-1, 144), axis=1)  # Flatten and find max
    max_rows, max_cols = np.unravel_index(max_indices, (12, 12))

    # Create masks to extract 3x3 windows around each seed cell
    row_indices = np.arange(-1, 2).reshape(1, -1, 1) + max_rows[:, None, None]
    col_indices = np.arange(-1, 2).reshape(1, 1, -1) + max_cols[:, None, None]

    # Clip indices to stay within bounds
    row_indices = np.clip(row_indices, 0, 11)
    col_indices = np.clip(col_indices, 0, 11)

    # Gather the 3x3 energies using advanced indexing
    E_3 = images[np.arange(images.shape[0])[:, None, None], row_indices, col_indices].sum(axis=(1, 2))

    # Calculate R9
    R9 = np.divide(E_3, E_total, where=E_total > 0, out=np.zeros_like(E_total))
    return R9



for particle in ['photon', 'pion']:

    images      = np.load(data_path + "train_img_" + particle + ".npy")
    conditions  = np.load(data_path + "train_conditions_" + particle + ".npy")


    # downsample
    images = downsample(images[:size])
    conditions = conditions[:size]

    # brightest pixel

    fig = plt.figure()

    plt.title("Comparison - R9")

    bin_centers = np.arange(0.5,1.3,0.02)
    bin_edges = get_binedges(bin_centers)

    for idx, noise_level in enumerate(noise_levels):

        noise = np.random.normal(loc = 0, scale = noise_level, size = images.shape)

        images = images + noise

        # calculate the shower shapes with the helper function in the beginning
        r9 = calculate_r9(images)
        hist_values, _ = np.histogram(r9, bin_edges)

        plt.step(bin_centers, hist_values, where='mid', color=colors[idx], lw = 3.5, label = f"Noise = {noise_level:.1f} MeV")
        plt.fill_between(bin_centers, hist_values, step='mid', color=colors[idx], alpha = 0.03)


    plt.xlabel(r"Energy [MeV]")
    plt.ylabel("# Events")

    plt.legend(loc="upper left")


    plt.tight_layout()
    fig.savefig("comparison_noise_r9_" + particle + ".png", dpi = 350)
