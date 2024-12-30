import torch
import torch.nn as nn
import numpy as np
import os


class SimData(torch.utils.data.Dataset):
    """ 
    Class for our custom dataset to be implemented with pytorchs DataLoader 
    Args:       root_dir - Directory of datasets
                particle - Specify the type of particle(s) in dataset
                mode - Specify purpose of dataset (training, validation or testing)
                transforms - The transformations on the dataset
    
    """

    def __init__(self, root_dir, particle, mode, transforms):

        assert particle in ("photon", "pion", "both"), "Please chose particle type to be 'photon', 'pion' or 'both'."
        assert mode in ("train", "test", "validation"), "Please chose mode to be 'train', 'test' or 'validation'."

        # if both set particle to "" for loading below
        if (particle == "both") : particle = ""

        self.root_dir               = root_dir
        self.transforms             = transforms
        self.mode                   = mode
        self.images                 = np.load(self.root_dir + self.mode + "_img_" + particle + ".npy")
        self.labels                 = np.load(self.root_dir + self.mode + "_labels_" + particle + ".npy")
        self.conditions             = np.load(self.root_dir + self.mode + "_conditions_" + particle + ".npy")

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
    """ Convert the numpy arrays to torch tensors. """

    def __call__(self, sample):

        images, labels, conditions, weights = sample['images'].copy(), sample['labels'].copy(), sample['conditions'].copy(), sample['weights'].copy()

        mod_sample = {'images' : torch.from_numpy(images), 'labels' : torch.from_numpy(np.array([labels])), 'conditions' : torch.from_numpy(conditions), 'weights' : torch.from_numpy(weights)}

        return mod_sample
    

class BinaryLabels(object):
    """ Converts the string labels of the images to a binary variable (0 for photon, 1 for pion). """

    def __call__(self, sample):

        labels = sample['labels'].copy()

        binary_labels = np.where(np.char.startswith(labels, 'pion'), 1, 0)

        sample['labels'] = binary_labels

        return sample
    
class DownSample(object):
    """ Downsamples the images from 24 x 24 to lower resolution by summing the values of neigboring pixels """

    def __init__(self, input_size, output_size):

        self.input_size = input_size

        assert input_size % output_size == 0, "New size should be divisor of old size (24). "
        self.output_size = output_size


    def __call__(self, sample):

        images = sample['images'].copy()

        # Calculate the block size for summation
        block_size = self.input_size // self.output_size
        
        reshaped_image = images.reshape(self.output_size, block_size, self.output_size, block_size)
        
        # Sum over the second and fourth dimensions (neighboring pixels)
        downsized_image = reshaped_image.sum(axis=(1, 3))
        
        sample['images'] = downsized_image

        return sample
    

class FlattenImage(object):
    """ Class for flattening the images as input of the normalizing flows """

    def __init__(self, n_features):

        self.n_features = n_features

    def __call__(self, sample):

        images = sample['images'].copy()
        images = images.reshape(self.n_features, 1)

        sample['images'] = images

        return sample
    


class NormalizeImage(object):
    """ A class for normalizing the image to [0,1]. """

    def __init__(self, noise : bool):

        self.epsilon  = 1e-8
        self.noise = noise

    def __call__(self, sample):

        images, conditions = sample['images'].copy(), sample['conditions'].copy()

        # add 1000 as 6 * std to make sure the values are still positive (deducted in postprocessing)
        if self.noise: images += (6 * conditions[3])


        # normalize images based on energy condition
        norm_images = np.divide(images, conditions[0])


        # little check if we still have zero value pixels
        if (np.sum(norm_images < 0) > 0): print("Alert, value(s):", norm_images[norm_images < 0])
        if (np.sum(norm_images > 1) > 0): print("Alert, value(s):", norm_images[norm_images > 1])



        # make sure its between 0 and 1
        #clipped_images = np.clip(norm_images, a_min = 0, a_max = 1)

        sample['images'] = norm_images

        return sample
    


def condition_scaling(condition, lower_bound, higher_bound, use_torch = True):
    ''' Scale the connditions of an abritatry range to a [-1, 1] range using logarithmic transformation '''

    to_subtract = lower_bound - 1
    new_higher_bound = higher_bound - to_subtract

    if use_torch:
        y = 2 * (torch.log(condition - to_subtract)/(torch.log(torch.Tensor([new_higher_bound])))) - 1
    else:
        y = 2 * (np.log(condition - to_subtract)/(np.log(new_higher_bound))) - 1

    return y



class LogitTransformation(object):
    """ Class for performing Logit transformation as in CaloFlow [https://arxiv.org/abs/2106.05285]"""

    def __init__(self, noise : bool, alpha = 6.7e-3):

        self.alpha = alpha
        self.noise = noise

    def __call__(self, sample):

        images, conditions = sample['images'].copy(), sample['conditions'].copy()

        # logit transformation of the images
        u = self.alpha + (1 - 2*self.alpha) * images
        u_logit = np.log(u / (1 - u))

        # logit transformation of the conditions
        energy_logit = condition_scaling(conditions[0]/1000, lower_bound = 20, higher_bound = 100, use_torch = False) # first convert to GeV
        shielding = condition_scaling(conditions[1], lower_bound=0.5, higher_bound=1.5, use_torch=False)
        distance = condition_scaling(conditions[2], lower_bound=50, higher_bound=90, use_torch=False)

        if self.noise: 
            std_logit = np.log10(conditions[3]/10)
            conditions_logit = np.array([energy_logit, shielding, distance, std_logit])
        else:
            conditions_logit = np.array([energy_logit, shielding, distance])




        sample['images'] = u_logit
        sample['conditions'] = conditions_logit

        return sample
    


class AddNoise(object):
    """ Class for performing Logit transformation as in CaloFlow [https://arxiv.org/abs/2106.05285]"""

    def __init__(self, active = True, noise_level = 1e-3, generator = None):

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
    ''' Class for adding weights depending on the value of the condition. '''

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

        # calculate the additional weight factor [0, 0.1]
        thickness_factor = (conditions_thickness - min_thickness) / (max_thickness + min_thickness) * self.max_weighting
        distance_factor = (conditions_distance - min_distance) / (max_distance + min_distance) * self.max_weighting

        # original weight
        weights = np.ones(shape = (1,)) 

        if self.active:
            weights += thickness_factor
            weights += distance_factor

            
        sample['weights'] = weights

        return sample

def postprocessing(sample, E_inc, sample_size, image_size_x, image_size_y, threshold, noise = None, alpha = 6.7e-3):
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

    if noise is not None:
        sample = sample - (6 * noise.view(-1,1))

    # reshape the image
    sample = sample.reshape(sample_size, image_size_x, image_size_y, 1)

    # cut out beneath threshold (if we did not train with noise)
    if noise is None:
        sample[sample < threshold] = 0

    return sample
