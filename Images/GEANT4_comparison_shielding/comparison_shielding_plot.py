import copy
from sklearn.cluster import DBSCAN
from scipy.ndimage import center_of_mass


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.CMS])

import numpy as np

data_path = "/net/scratch_cms3a/kann/fast_calo_flow/data/shielding/"




size = 60_000
shielding_ranges = np.array([(0.5, 0.75), (0.75,1.0), (1., 1.25), (1.25, 1.50)])
colors = ['purple', 'navy', 'mediumseagreen', 'darkorange']



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
    distances = np.sqrt(delta_x**2 + delta_y**2) 

    # get shower width
    numerator = np.sum(distances*images, axis = (1,2))
    shower_width = numerator / total_energy

    return shower_width


def calculate_r9(images, image_size_x, image_size_y):
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
    max_indices = np.argmax(images.reshape(-1, image_size_x*image_size_y), axis=1)  # Flatten and find max
    max_rows, max_cols = np.unravel_index(max_indices, (image_size_x, image_size_y))

    # Create masks to extract 3x3 windows around each seed cell
    row_offsets = np.arange(-1, 2).reshape(1, -1, 1)  # [-1, 0, 1] for rows
    col_offsets = np.arange(-1, 2).reshape(1, 1, -1)  # [-1, 0, 1] for columns
    row_indices = np.clip(max_rows[:, None, None] + row_offsets, 0, image_size_x - 1)  # Clip to valid row range
    col_indices = np.clip(max_cols[:, None, None] + col_offsets, 0, image_size_y - 1)  # Clip to valid col range


    # Clip indices to stay within bounds
    row_indices = np.clip(row_indices, 0, 11)
    col_indices = np.clip(col_indices, 0, 11)

    # Gather the 3x3 energies using advanced indexing
    E_3 = images[np.arange(images.shape[0])[:, None, None], row_indices, col_indices].sum(axis=(1, 2))

    # Calculate R9
    R9 = np.divide(E_3, E_total, where=E_total > 0, out=np.zeros_like(E_total))
    return R9


def sigma_ieta_ieta(images):

    ''' Calculate sigma_ieta_ieta for images'''


    # get shapes
    n, image_size_x, image_size_y = images.shape
    
    # Find the flat index of the maximum pixel for each image
    flat_max_indices = np.argmax(images.reshape(n, -1), axis=1)
    max_x, max_y = np.unravel_index(flat_max_indices, shape = (image_size_x, image_size_y))
    
    # Create meshgrid for offsets relative to the center (range -2 to +2)
    offsets = np.arange(-2, 3)
    dx, dy = np.meshgrid(offsets, offsets, indexing='ij')
    
    # Add offsets to max coordinates (with boundary clipping)
    x_indices = np.clip(max_x[:, None, None] + dx, 0, image_size_x - 1)
    y_indices = np.clip(max_y[:, None, None] + dy, 0, image_size_y - 1)
    
    # Gather the 5x5 cutouts
    lr_cut = images[np.arange(n)[:, None, None], x_indices, y_indices]


    weights_LR = np.maximum( 0, 4.7 + np.log(lr_cut / lr_cut.sum(axis=(1,2), keepdims = True)))
    eta_grid_LR = np.mgrid[0:5, 0:5][1] * (0.022/1.29) # (0.022/1.29) is crystal size in eta  
    eta_grid_LR = np.repeat(eta_grid_LR[None, :, :], n, axis = 0)


    eta_bar = np.array([center_of_mass(lr_image)[1] for lr_image in lr_cut])

    sigma_ietaieta_LR = np.sum( weights_LR * (eta_grid_LR - eta_bar[:, None, None] * (0.022/1.29))**2 , axis = (1,2)) / np.sum(weights_LR, axis = (1,2))

    return sigma_ietaieta_LR


def difference_brigtest_x_brightest(images, x):

    ''' Get the spatial difference of the brightest and x-th brightest pixel.'''

    assert x > 1, "Value of x must be bigger than 1."

    # location of maximum
    max_x, max_y = np.unravel_index(np.argmax(images.reshape(-1,16*8), axis = 1), (16,8))

    for i in range(1, x):

        brightest_pixels = np.max(images, axis = (1,2))
        # mask out maximum
        images = np.where(images == brightest_pixels[:, None, None], -np.inf, images)

    # location of second brightest pixel
    max2_x, max2_y = np.unravel_index(np.argmax(images.reshape(-1,16*8), axis = 1), (16,8))

    distances = np.sqrt( (max2_x -max_x)**2 + (max2_y - max_y)**2 )

    return distances

particle = 'photon'
image_size_x = 16
image_size_y = 8

images      = np.load(data_path + "train_img_" + particle + ".npy")[:size]
conditions  = np.load(data_path + "train_conditions_" + particle + ".npy")[:size]


# subselect number of samples
# images      = images[:size]
# conditions  = conditions[:size]


fig = plt.figure()

plt.title(r"Comparison - Test")

bin_centers = np.arange(0.00002,0.0009,0.00001)
bin_edges = get_binedges(bin_centers)

for idx, shielding_range in enumerate(shielding_ranges):

    mask = (conditions[:,1] > shielding_range[0]) & (conditions[:,1] < shielding_range[1])
    print("Check")
    images_masked = images[mask]
    conditions_masked = conditions[mask]

    # # calculate the shower shapes with the helper function in the beginning
    
    # max_x, max_y = np.unravel_index(np.argmax(images_masked.reshape(-1,16*8), axis = 1), (16,8))
    # barycenter_x, barycenter_y = get_barycenter(images_masked)
    # center_y, center_x = 17/2, 9/2

    # distance_barycenter = np.sqrt((barycenter_x - max_x)**2 + (barycenter_y -max_y)**2)

    sigma_ieta_ieta_values = sigma_ieta_ieta(images_masked)

    hist_values, _ = np.histogram(sigma_ieta_ieta_values, bin_edges)

    plt.step(bin_centers, hist_values, where='mid', color=colors[idx], lw = 3.5, label = f"d $\\in ({shielding_range[0]},{shielding_range[1]}) X_0$")
    plt.fill_between(bin_centers, hist_values, step='mid', color=colors[idx], alpha = 0.03)


plt.xlabel(r"Distance [Crystal Width]")
plt.ylabel("# Events")

plt.legend(loc="upper right")


plt.tight_layout()
fig.savefig("comparison_shielding_test" + particle + ".png", dpi = 350)
