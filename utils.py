''' Just some leftover classes & functions that don't fit in any other file ....'''


import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import center_of_mass

def get_binedges(bin_centers):

    bin_edges = np.concatenate((
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[1:] + bin_centers[:-1]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]
    ))

    return bin_edges

### Functions for computing the barycenter and the shower width

def get_barycenter(images):

    images = images.squeeze()

    total_energy    = np.sum(images, axis=(1,2))

    x_coords, y_coords = np.meshgrid(np.arange(images.shape[2]), np.arange(images.shape[1]))

    # Expand dimensions to match sample size
    x_coords = np.broadcast_to(x_coords, images.shape)
    y_coords = np.broadcast_to(y_coords, images.shape)

    barycenter_x    =  np.sum(x_coords * images, axis=(1, 2)) / total_energy
    barycenter_y    =  np.sum(y_coords * images, axis=(1, 2)) / total_energy

    return barycenter_x, barycenter_y

    # scipy center of mass function

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

    images = images.squeeze()

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
    
    # Get the 5x5 cutouts
    lr_cut = images[np.arange(n)[:, None, None], x_indices, y_indices]

    epsilon = 1e-8 # add smal epsilon to avoid log(0)

    #### calculate the sigma_ieta_ieta ####
    weights_LR = np.maximum( 0, 4.7 + np.log((lr_cut + epsilon)/ lr_cut.sum(axis=(1,2), keepdims = True))) # weights, based on the energy
    eta_grid_LR = np.mgrid[0:5, 0:5][0] * (0.022/1.29) # (0.022/1.29) is crystal size in eta  
    eta_grid_LR = np.repeat(eta_grid_LR[None, :, :], n, axis = 0)


    eta_bar = np.array([center_of_mass(lr_image)[1] for lr_image in lr_cut])

    sigma_ietaieta_LR = np.sum( weights_LR * (eta_grid_LR - eta_bar[:, None, None] * (0.022/1.29))**2 , axis = (1,2)) / np.sum(weights_LR, axis = (1,2))

    return sigma_ietaieta_LR


def distance_brigtest_x_brightest(images, x, image_size_x, image_size_y):

    ''' Get the spatial difference of the brightest and x-th brightest pixel.'''

    assert x > 1, "Value of x must be bigger than 1."

    # location of maximum
    max_x, max_y = np.unravel_index(np.argmax(images.reshape(-1,image_size_x*image_size_y), axis = 1), (image_size_x,image_size_y))

    for i in range(1, x):

        brightest_pixels = np.max(images, axis = (1,2))
        # mask out maximum
        images = np.where(images == brightest_pixels[:, None, None], -np.inf, images)

    # location of second brightest pixel
    max2_x, max2_y = np.unravel_index(np.argmax(images.reshape(-1,image_size_x*image_size_y), axis = 1), (image_size_x,image_size_y))

    distances = np.sqrt( (max2_x -max_x)**2 + (max2_y - max_y)**2 )

    return distances

### Class for early stopping

class EarlyStopper:
    '''
    Class for early stopping when validation loss didn't improve over a specified number of epochs
    '''
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf # initial value

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss  # set new minimum
            self.counter = 0                            # reset counter
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1                           # else increase counter
            if self.counter >= self.patience:           # if counter surpasses patiences stopp training
                return True
        return False



def load_test_data(data_path, sample_size, particle_type, detector_noise_level, shielding_range, distance_range, threshold):


    # load data, depending on which particle to analyse
    if (particle_type == 'both'):

        # load images and conditions
        data_img_photon = np.load(data_path + "test_img_" + 'photon' + ".npy")
        data_img_pion = np.load(data_path + "test_img_" + 'pion' + ".npy")

        data_conditions_photon = np.load(data_path + 'test_conditions_photon.npy')
        data_conditions_pion = np.load(data_path + 'test_conditions_pion.npy')


        ## set threshold for sparsity plots
        data_img_photon[data_img_photon < threshold] = 0
        data_img_pion[data_img_pion < threshold] = 0


        # this is a check if we want to evaluate a specific noise level, then add that noise level onto test set
        if (detector_noise_level):

            # sample gaussian noise
            noise_photon    = np.random.normal(loc = 0, scale = detector_noise_level, size = data_img_photon.shape)
            noise_pion      = np.random.normal(loc = 0, scale = detector_noise_level, size = data_img_pion.shape)

            noise_condition_photon = np.full(shape = (data_img_photon.shape[0],1), fill_value=detector_noise_level)
            noise_condition_pion   = np.full(shape = (data_img_pion.shape[0],1), fill_value=detector_noise_level)

            data_conditions_photon = np.concatenate((data_conditions_photon, noise_condition_photon), axis = 1)
            data_conditions_pion   = np.concatenate((data_conditions_pion, noise_condition_pion), axis = 1)

            # add noise
            data_img_photon = data_img_photon + noise_photon
            data_img_pion   = data_img_pion + noise_pion

        if (shielding_range != (0.5, 1.5)):

            lower_bound, higher_bound = shielding_range[0], shielding_range[1]

            mask_photon = (data_conditions_photon[:,1] > lower_bound) & (data_conditions_photon[:,1] < higher_bound)
            mask_pion   = (data_conditions_pion[:,1] > lower_bound) & (data_conditions_pion[:,1] < higher_bound)

            data_img_photon = data_img_photon[mask_photon]
            data_img_pion   = data_img_pion[mask_pion]

            data_conditions_photon  = data_conditions_photon[mask_photon]
            data_conditions_pion    = data_conditions_pion[mask_pion]

        if (distance_range != (50, 90)):

            lower_bound, higher_bound = distance_range[0], distance_range[1]

            mask_photon = (data_conditions_photon[:,2] > lower_bound) & (data_conditions_photon[:,2] < higher_bound)
            mask_pion   = (data_conditions_pion[:,2] > lower_bound) & (data_conditions_pion[:,2] < higher_bound)

            data_img_photon = data_img_photon[mask_photon]
            data_img_pion   = data_img_pion[mask_pion]

            data_conditions_photon  = data_conditions_photon[mask_photon]
            data_conditions_pion    = data_conditions_pion[mask_pion]


        data_img = list([data_img_photon, data_img_pion]) # but data in a list
        data_conditions = list([data_conditions_photon, data_conditions_pion])

    else:

        data_img = np.load(data_path + "test_img_" + particle_type + ".npy")
        data_conditions = np.load(data_path + "test_conditions_" + particle_type + ".npy")

        ## set threshold for sparsity plots
        data_img[data_img < threshold] = 0


        # this is a check if we want to evaluate a specific noise level, then add that noise level onto test set
        if (detector_noise_level):

            # sample gaussian noise
            noise    = np.random.normal(loc = 0, scale = detector_noise_level, size = data_img.shape)

            noise_condition = np.full(shape = (data_img.shape[0],1), fill_value=detector_noise_level)

            data_conditions = np.concatenate((data_conditions, noise_condition), axis = 1)

            # add noise
            data_img = data_img + noise


        if (shielding_range != (0.5,1.5)):

            lower_bound, higher_bound = shielding_range[0], shielding_range[1]

            mask = (data_conditions[:,1] >= lower_bound) & (data_conditions[:,1] <= higher_bound)

            data_img = data_img[mask]

            data_conditions  = data_conditions[mask]

        if (distance_range != (50, 90)):

            lower_bound, higher_bound = distance_range[0], distance_range[1]

            mask = (data_conditions[:,2] > lower_bound) & (data_conditions[:,2] < higher_bound)

            data_img = data_img[mask]

            data_conditions  = data_conditions[mask]


        data_img = list([data_img])
        data_conditions = list([data_conditions])

    
    


    # select subset (according to sample size) of data for all particles
    data = [subset[:sample_size] for subset in data_img]
    conditions = [subset[:sample_size] for subset in data_conditions]


    return data, conditions