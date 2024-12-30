''' 
Additional functions and classes utilized for training, evaluation or data processing.
'''


import numpy as np
from scipy.ndimage import center_of_mass


def get_binedges(bin_centers):

    ''' Function for calculating bin_edges from bin_centers.'''

    bin_edges = np.concatenate((
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[1:] + bin_centers[:-1]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]
    ))

    return bin_edges


def get_barycenter(images):

    ''' Function for getting the energy barycenter from the images. [Alternative: scipy.ndimage.center_of_mass]'''

    images = images.squeeze()

    total_energy    = np.sum(images, axis=(1,2))

    x_coords, y_coords = np.meshgrid(np.arange(images.shape[2]), np.arange(images.shape[1]))

    # Expand dimensions to match sample size
    x_coords = np.broadcast_to(x_coords, images.shape)
    y_coords = np.broadcast_to(y_coords, images.shape)

    barycenter_x    =  np.sum(x_coords * images, axis=(1, 2)) / total_energy
    barycenter_y    =  np.sum(y_coords * images, axis=(1, 2)) / total_energy

    return barycenter_x, barycenter_y

def shower_width(images):

    ''' Function for calculating the shower width of the shower images. '''

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

    ''' Calculate R9 values for a batch of shower images.  '''

    
    # Total energy of each image
    E_total = images.sum(axis=(1, 2))


    # Find the indices of the maximum energy crystal for each image
    max_indices = np.argmax(images.reshape(-1, image_size_x*image_size_y), axis=1)
    max_rows, max_cols = np.unravel_index(max_indices, (image_size_x, image_size_y))

    # create masks to extract 3x3 windows around the maximum
    row_offsets = np.arange(-1, 2).reshape(1, -1, 1)
    col_offsets = np.arange(-1, 2).reshape(1, 1, -1)
    row_indices = np.clip(max_rows[:, None, None] + row_offsets, 0, image_size_x - 1)  # Clip to valid row range
    col_indices = np.clip(max_cols[:, None, None] + col_offsets, 0, image_size_y - 1)  # Clip to valid col range

    # Sum the 3x3 energies around maximum
    E_3 = images[np.arange(images.shape[0])[:, None, None], row_indices, col_indices].sum(axis=(1, 2))

    # Calculate R9
    R9 = np.divide(E_3, E_total, where=E_total > 0, out=np.zeros_like(E_total))
    return R9


def sigma_ieta_ieta(images):

    ''' Calculate the sigma_ieta_ieta from https://arxiv.org/pdf/2012.06888. DISCLAIMER: Currently something wrong. '''

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

    ''' Get the euclidean distance of the brightest and x-th brightest pixel.'''

    assert x > 1, "Value of x must be bigger than 1."

    # location of maximum
    max_x, max_y = np.unravel_index(np.argmax(images.reshape(-1,image_size_x*image_size_y), axis = 1), (image_size_x,image_size_y))

    # iteratively find the xth brightest 
    for _ in range(1, x):

        brightest_pixels = np.max(images, axis = (1,2))
        
        images = np.where(images == brightest_pixels[:, None, None], -np.inf, images) # mask out maximum

    # location of second brightest pixel
    max2_x, max2_y = np.unravel_index(np.argmax(images.reshape(-1,image_size_x*image_size_y), axis = 1), (image_size_x,image_size_y))

    distances = np.sqrt( (max2_x -max_x)**2 + (max2_y - max_y)**2 )

    return distances


class EarlyStopper:

    ''' Class for early stopping when validation loss didn't improve over a specified number of epochs. '''

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
