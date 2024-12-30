import torch
import torch.nn as nn
import numpy as np
import os


class SimData(torch.utils.data.Dataset):
    ''' 
    Class for our custom dataset to be implemented with pytorchs DataLoader 
    Args:       data_dir - Directory of datasets
                mode - Specify purpose of dataset (training, validation or testing)
                transforms - The transformations on the dataset
    
    '''

    def __init__(self, data_dir, mode, transforms):

        assert mode in ("train", "test", "validation"), "Please chose mode to be 'train', 'test' or 'validation'."

        self.data_dir               = data_dir
        self.transforms             = transforms
        self.mode                   = mode
        self.images                 = np.load(self.data_dir + self.mode + "_img_" + ".npy")
        self.labels                 = np.load(self.data_dir + self.mode + "_labels_" + ".npy")
        self.conditions             = np.load(self.data_dir + self.mode + "_conditions_" + ".npy")

    def __len__(self):
        return len(self.conditions)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        sample_images      = self.images[idx]
        sample_labels      = self.labels[idx]
        sample_conditions  = self.conditions[idx]

        # get data in dictionaries
        sample = {'images' : sample_images, 'labels' : sample_labels, 'conditions' : sample_conditions}

        # preform transformations on the samples
        sample = self.transforms(sample)

        return sample


class ToTensor(object):

    ''' Convert the numpy arrays to torch tensors. '''

    def __call__(self, sample):

        images, labels, conditions, weights = sample['images'].copy(), sample['labels'].copy(), sample['conditions'].copy(), sample['weights'].copy()

        mod_sample = {'images' : torch.from_numpy(images), 'labels' : torch.from_numpy(np.array([labels])), 'conditions' : torch.from_numpy(conditions), 'weights' : torch.from_numpy(weights)}

        return mod_sample
    

class FlattenImage(object):

    ''' Class for flattening the images as input of the normalizing flows. '''

    def __init__(self, n_features):

        self.n_features = n_features

    def __call__(self, sample):

        images = sample['images'].copy()
        images = images.reshape(self.n_features, 1)

        sample['images'] = images

        return sample
    


class NormalizeImage(object):

    ''' A class for normalizing the image to [0,1]. '''

    def __init__(self):

        self.epsilon  = 1e-8

    def __call__(self, sample):

        images, conditions = sample['images'].copy(), sample['conditions'].copy()

        # normalize images based on energy condition
        norm_images = np.divide(images, conditions[0])

        sample['images'] = norm_images

        return sample
    


def condition_scaling(condition, lower_bound, higher_bound, use_torch = True):

    ''' Scale the conditions of an abritatry range to a [-1, 1] range using logarithmic transformation '''

    to_subtract = lower_bound - 1
    new_higher_bound = higher_bound - to_subtract

    # to cases wether calculation is performed using torch or numpy
    if use_torch:
        y = 2 * (torch.log(condition - to_subtract)/(torch.log(torch.Tensor([new_higher_bound])))) - 1
    else:
        y = 2 * (np.log(condition - to_subtract)/(np.log(new_higher_bound))) - 1

    return y



class LogitTransformation(object):

    ''' Class for performing Logit transformation as in CaloFlow [https://arxiv.org/abs/2106.05285]'''

    def __init__(self, alpha = 1e-6):

        self.alpha = alpha

    def __call__(self, sample):

        images, conditions = sample['images'].copy(), sample['conditions'].copy()

        # logit transformation of the images
        u = self.alpha + (1 - 2 * self.alpha) * images
        u_logit = np.log(u / (1 - u))

        # logit transformation of the conditions (energy first scaled to GeV)
        energy_logit = condition_scaling(conditions[0]/1000, lower_bound = 20, higher_bound = 100, use_torch = False) # first convert to GeV
        thickness = condition_scaling(conditions[1], lower_bound=0.5, higher_bound=1.5, use_torch=False)
        distance = condition_scaling(conditions[2], lower_bound=50, higher_bound=90, use_torch=False)
                                     
        conditions_logit = np.array([energy_logit, thickness, distance])


        sample['images'] = u_logit
        sample['conditions'] = conditions_logit

        return sample
    


class AddNoise(object):

    ''' Class adding low energy noise as in CaloFlow [https://arxiv.org/abs/2106.05285]. '''

    def __init__(self, active = True, noise_level = 1e-6, generator = None):

        self.active = active
        self.noise_level = noise_level
        self.generator = generator

    def __call__(self, sample):

        images = sample['images'].copy()

        noise = self.generator.uniform(low = 0, high = self.noise_level, size = (images.shape))

        if self.active: images += noise

        sample['images'] = images

        return sample
    

    
class AddWeights(object):
    ''' Class for adding weights depending on the values of the conditions. DISCLAIMER: Currently not used. '''

    def __init__(self, active = False, max_weighting = 0.05):

        self.active = active
        self.max_weighting = max_weighting

    def __call__(self, sample):

        

        conditions = sample['conditions'].copy()

        conditions_thickness = conditions[1]
        conditions_distance  = conditions[2]

        min_thickness = 0.5
        max_thickness = 1.5

        min_distance  = 50
        max_distance  = 90

        # calculate the additional weight factor [0, 0.1] for low values of the condition
        thickness_factor    = ( (conditions_thickness - min_thickness) / (max_thickness + min_thickness) ) * self.max_weighting
        distance_factor     = ( (conditions_distance - min_distance) / (max_distance + min_distance) ) * self.max_weighting

        # original weights
        weights = np.ones(shape = (1,)) 

        if self.active:
            weights += thickness_factor
            weights += distance_factor

            
        sample['weights'] = weights

        return sample


def postprocessing(sample, E_inc, sample_size, image_size_x, image_size_y, threshold, alpha = 1e-6):

    ''' Function for postprocessing the averages '''


    # reverse logit transformation
    sample = torch.exp(sample)/(1+torch.exp(sample))

    # reverse alpha transformation
    sample = ( sample - alpha ) / ( 1 - 2 * alpha)

    # make sure really in [0,1]
    sample = torch.clamp(sample, min = 0, max = 1)

    # # renormalize:
    # sum_energy = torch.sum(sample, dim = (1,)).squeeze()
    # sample = sample / sum_energy[:, None]

    # reverse normalization
    sample = sample * ( E_inc.view(-1,1) )

    # reshape the image
    sample = sample.reshape(sample_size, image_size_x, image_size_y, 1)

    # cut out beneath threshold (because of noise applied in preprocessing)
    sample[sample < threshold] = 0

    return sample


def load_test_data(data_path, sample_size, thickness_range, distance_range, threshold):

    ''' Function for loading the data of specified size and possibly specified parameter range. '''

    data_img = np.load(data_path + "test_img_" + ".npy")
    data_conditions = np.load(data_path + "test_conditions_" + ".npy")

    ## set threshold for sparsity plots
    data_img[data_img < threshold] = 0

    # subselect specified thickness range
    lower_bound, higher_bound = thickness_range[0], thickness_range[1]

    mask_thickness = (data_conditions[:,1] >= lower_bound) & (data_conditions[:,1] <= higher_bound)

    data_img = data_img[mask_thickness]
    data_conditions  = data_conditions[mask_thickness]

    # subselect specified thickness range
    lower_bound, higher_bound = distance_range[0], distance_range[1]

    mask_distance = (data_conditions[:,2] >= lower_bound) & (data_conditions[:,2] <= higher_bound)

    data_img = data_img[mask_distance]
    data_conditions  = data_conditions[mask_distance]

    assert sample_size <= len(data_img), "Specified sample size is bigger than GEANT4 testset."

    # select subset (according to sample size) of data
    data = data_img[:sample_size]
    conditions = data_conditions[:sample_size]


    return data, conditions
