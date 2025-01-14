'''
This file contains the helper functions and classes for the evaluation work of the models and is called in main.py
It contains functions for plotting histograms with relevant features, sample images, averages and correlations.
'''

# Used libraries
import os
import copy
import yaml
import time

import torchvision
import torch
import torch.nn as nn

import zuko

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.CMS])


# Load the custom module
from custom_model.model import NSF

### Function & Classes from other files
from data import SimData, LogitTransformation, AddNoise, NormalizeImage, FlattenImage, ToTensor, postprocessing, condition_scaling, AddWeights, load_test_data
from classifier import train_classifier, get_data, plot_history, plot_roc, plot_learning_rates, ElectromagneticShowerClassifier
from plotting import plot_histogram, plot_2d_parameterspace
from utils import get_barycenter, shower_width, calculate_r9, distance_brigtest_x_brightest, sigma_ieta_ieta

    
### Main class for the heavy work... Our model Judge is judging the goodness of our generator
class Judge:

    def __init__(self, run_id : str, sample_size : int, data_path : str, device : str, parameters : dict, thickness_bin = None , distance_bin = None, bin_comparison_thickness = False, bin_comparison_distance = False, generate = True):

        ''' 
        Initialize the framework for evaluation, reads in the data and generates samples. 
        Args:   run_id  -  Specifies the run / model to evaluate,
                sample_size  -  Number of samples to evaluate/generate,
                data_path  -  Directory of data files,
                device  -- Device on which we want to evaluate,
                parameters -- Dictionary with parameters of model,
                thickness_bin -- Integer or None, specifies if model should be evaluated on specific range of thickness,
                distance_bin -- Integer or None, specifies if model should be evaluated on specific range of distance,
                bin_comparison_thickness -- Boolean, if comparison between the ranges shall be plotted,
                bin_comparison_distance -- Boolean, if comparison between the ranges shall be plotted,
                generate -- Boolean, if samples should be generated or loaded.

        '''

        self.run_id = run_id
        self.sample_size = sample_size
        self.data_path = data_path
        self.device = device

        # seed for random number generator
        self.torch_seed = 42


        # evaluate on specific conditions if given
        self.thickness_bin              = thickness_bin
        self.distance_bin               = distance_bin
        self.bin_comparison_thickness   = bin_comparison_thickness
        self.bin_comparison_distance    = bin_comparison_distance

        # option for evaluation of specific thickness thickness
        assert (not thickness_bin) or (thickness_bin in [1,2,3,4]), "The bin for the thickness values must be an integer between 1 and 4."
        assert (not distance_bin)  or (distance_bin in [1,2,3,4]), "The bin for the distance values must be an integer between 1 and 4."
        assert (not thickness_bin and not distance_bin) or (sample_size <= 60_000), "When selecting a specific bin for evaluation, sample size must not exceed 60 000."


        # bins for thickness
        thickness_ranges        = [(0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25,1.5)]
        self.thickness_range    = (0.5,1.5) if not thickness_bin else thickness_ranges[thickness_bin-1]

        # bins for distance
        distance_ranges         = [(50,60), (60,70), (70,80), (80,90)]
        self.distance_range     = (50,90) if not distance_bin else distance_ranges[distance_bin-1]


        ################ GET PARAMETERS FROM YAML ################
        

        # path under which the config.yml is saved
        self.config_path = f"/net/data_cms3a-1/kann/fast_calo_flow/results/{self.run_id}/"


        # read configuration of specified run for model and evaluation
        with open(self.config_path + 'config.yml', 'r') as file:
            parameters = yaml.safe_load(file)
        self.parameters = parameters


        # get preprocessing parameters form yaml file
        self.alpha = parameters['alpha']
        self.noise = parameters['noise']
        self.noise_level = parameters['noise_level']
        self.image_size_x = parameters['image_size_x']
        self.image_size_y = parameters['image_size_y']


        ##################### LOADING MODEL #####################

        
        # load best model from specified run (instantiate from parameters from yaml file and then load state dict)
        model = NSF(features = parameters['n_features'], context = parameters['n_conditions'], bins = parameters['n_bins'], transforms = parameters['n_transforms'], randperm = parameters['random_perm'], hidden_features = [parameters['n_aux_nodes']] * parameters['n_aux_layers'], p_dropout = parameters['p_dropout'])
        model.load_state_dict(torch.load(f"/net/data_cms3a-1/kann/fast_calo_flow/results/{self.run_id}/models/best_model.pt")['model'])
        model = model.eval() # set model to evaluation mode (only necessary if drop out or batch norm are active)
        self.model = model.to(self.device)


        ################ SET UP RESULT DIRECTORIES ###############


        # get evaluation path (path to store evaluation files)
        self.result_path = f"/net/data_cms3a-1/kann/fast_calo_flow/results/{self.run_id}/evaluation/"

        # if we specified a specific parameter value or range to investigate, we create a subfolder and make this our new result path 
        if (self.thickness_bin):
            os.makedirs(self.result_path + f"thickness_{int(self.thickness_bin)}/", exist_ok=True)
            self.result_path = self.result_path + f"thickness_{int(self.thickness_bin)}/"

        if (self.distance_bin):
            os.makedirs(self.result_path + f"distance_{int(self.distance_bin)}/", exist_ok=True)
            self.result_path = self.result_path + f"distance_{int(self.distance_bin)}/"

        if (self.bin_comparison_thickness):
            os.makedirs(self.result_path + f"bin_comparison_thickness/", exist_ok=True)

        if (self.bin_comparison_distance):
            os.makedirs(self.result_path + f"bin_comparison_distance/", exist_ok=True)


        # Setup plotting colors

        if (not thickness_bin and not distance_bin):
            self.plot_color = 'cornflowerblue'
            self.error_color = 'blueviolet'
        elif (thickness_bin):
            self.plot_color = ['orange', 'red', 'royalblue', 'darkmagenta'][self.thickness_bin - 1]
            self.error_color = ['orangered', 'darkred', 'navy', 'indigo'][self.thickness_bin - 1]
        else:
            self.plot_color = ['cadetblue', 'greenyellow', 'mediumpurple', 'royalblue'][self.distance_bin - 1]
            self.error_color = ['teal', 'yellowgreen', 'rebeccapurple', 'darkblue'][self.distance_bin - 1]


        ################### LOAD GEANT4 DATA ####################

        self.data, self.conditions = load_test_data(data_path = data_path, sample_size = self.sample_size, thickness_range = self.thickness_range, distance_range = self.distance_range, threshold = self.noise_level)



        ###################### GET SAMPLES ######################

        # Either load samples if generate is specified & samples already exist
        if (not generate) and ( os.path.isfile(self.result_path + "data/samples.npy") and os.path.isfile(self.result_path + "data/conditions.npy") ):


            self.samples = np.load(self.result_path + "data/samples.npy")
            self.sample_conditions = np.load(self.result_path + "data/conditions.npy")

            if len(self.samples) < self.sample_size: print("Issue: Loaded sample size is smaller than specified sample size.")

            print("Loaded dataset of size: ", len(self.samples))

            self.samples = self.samples[:self.sample_size]
            self.sample_conditions = self.sample_conditions[:self.sample_size]

            return None

        # Or sample images:

        ################ CONDITIONS FOR SAMPLING ################

        ### 1. ENERGY

        # energies for which we want to sample (as in the dataset of GEANT4)
        uniform_energies = np.random.uniform(20_000, 100_000, size = (self.sample_size,1))
        uniform_energies = torch.from_numpy(uniform_energies)

        ### 2. THICKNESS

        # thickness for which we want to sample (as in the dataset of GEANT4)
        uniform_thickness = np.random.uniform(*self.thickness_range, size = (self.sample_size,1))
        uniform_thickness = torch.from_numpy(uniform_thickness)

        ### 3. DISTANCE

        # distances for which we want to sample (as in the dataset of GEANT4)
        uniform_distance  = np.random.uniform(*self.distance_range, size = (self.sample_size,1))
        uniform_distance  = torch.from_numpy(uniform_distance)


        self.sample_conditions = torch.cat((uniform_energies, uniform_thickness, uniform_distance), dim = 1).cpu().numpy()            
        ### Note: We need the self.sample_conditions for the classifier


        ######################## SAMPLING ########################    

        # start timer for benchmarking purposes
        start_time = time.time()

        
        # preprocess conditions as during training
        conditions_energy       = condition_scaling(uniform_energies / 1000, lower_bound=20, higher_bound=100, use_torch = True).to(dtype=torch.float32).to(self.device)
        conditions_thickness    = condition_scaling(uniform_thickness, lower_bound=0.5, higher_bound=1.5, use_torch= True).to(dtype=torch.float32).to(self.device)
        conditions_distance     = condition_scaling(uniform_distance, lower_bound = 50.0, higher_bound=90, use_torch=True).to(dtype=torch.float32).to(self.device)


        conditions = torch.cat((conditions_energy, conditions_thickness, conditions_distance), dim = 1)


        ### Generate samples for the given conditions

        samples = self.model(conditions).sample()
        samples = samples.to(dtype=torch.float32)


        print("Number of nan-values:", np.sum(np.isnan(samples.cpu().numpy())))


        # Postprocessing of generated samples
        samples = torch.nan_to_num(samples, nan=0.0) # there are rarely nan values in the generated images
        samples = postprocessing(samples, uniform_energies.to(self.device), sample_size=self.sample_size, image_size_x=self.image_size_x, image_size_y=self.image_size_y, threshold= 1.0 * self.noise_level, alpha = self.alpha)
        samples = samples.cpu().to(dtype=torch.float32).numpy()
        self.samples = samples

        os.makedirs(self.result_path + "data/", exist_ok=True)
        np.save(self.result_path + "data/samples.npy", self.samples)
        np.save(self.result_path + "data/conditions.npy", self.sample_conditions)
        

        # Stop timer and output generation time for checking purposes
        end_time = time.time()
        print(f"Benchmarking - Duration of sampling and postprocessing {self.sample_size} images:   {end_time - start_time} s")
        

    @torch.no_grad()
    def calculate_testloss(self):

        ''' Function for calculationg the test loss of the model on the whole dataset. '''

        # set model to evaluation
        self.model.eval()

        # Random number generator for preprocessing
        generator = np.random.default_rng(self.torch_seed)

        # Transformation chain as in training
        transforms = torchvision.transforms.Compose([
                                        AddNoise(active=self.noise, noise_level = self.noise_level, generator=generator), 
                                        FlattenImage(self.image_size_x*self.image_size_y), 
                                        NormalizeImage(), 
                                        LogitTransformation(alpha = self.alpha),
                                        AddWeights(active = self.parameters['weights']),
                                        ToTensor()])
        # Load dataset
        dataset_test           = SimData(root_dir = self.data_path, mode = "test", transforms = transforms)

        # Instantiate dataloader
        dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=512, shuffle=True)

        # variable for loss each batch
        cummulative_test_loss = 0

        # Iteration for each batch
        for i, batch in enumerate(dataloader_test):

            images, conditions, weights = batch['images'], batch['conditions'], batch['weights']
                
            # send images and conditions to correct data type & device
            images = torch.squeeze(images).to(self.device).to(dtype = torch.float32)
            conditions = conditions.to(self.device).to(dtype = torch.float32)

            # evaluate the test_loss
            test_loss = -self.model(conditions).log_prob(images) * weights.to(self.device)
            cummulative_test_loss += test_loss.mean()

        # get test loss per batch from cummulative loss
        test_loss = cummulative_test_loss / (i+1)
   

        # Output
        print("-"*100)
        print(f"\033[1;31mTest loss: {test_loss:.3f}\033[0m")


    def plot_averages(self):

        ''' Function for plotting average shower images and their differences. '''

        # get averages (not considering any nan values)
        average_samples = np.nanmean(self.samples, axis=0).squeeze()
        average_data    = np.nanmean(self.data, axis=0).squeeze()

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols = 3, width_ratios = [3.5,4.5,4.5], figsize = (18,8), sharey = True)

        title = "Average~Images"
        fig.suptitle("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$", c="black", fontsize = 35, ha = 'left', va = 'top', x = 0.1, y = 1.02)


        cmap = copy.copy(matplotlib.colormaps["viridis"])
        cmap.set_under('w')

        # Use LogNorm to apply logarithmic scaling to color mapping
        norm = mcolors.LogNorm(vmin=1, vmax=100e3)


        # Samples
        im = ax1.imshow(average_samples, norm=norm, cmap=cmap)
        ax1.set_title("ParaFlow", c="navy", fontweight = 'bold')

        ax1.set_xticks([])
        ax1.set_yticks([])

        # GEANT4
        im = ax2.imshow(average_data, norm=norm, cmap=cmap)
        ax2.set_title("MC", c="navy", fontweight = 'bold')

        cbar = fig.colorbar(im, ax=ax2, extend = 'min')
        cbar.set_label('Energy [MeV]')

        ax2.set_xticks([])
        ax2.set_yticks([])


        # Difference in %
        relative_difference = (average_data - average_samples)/average_data * 100

        im = ax3.imshow(relative_difference, vmin=-15, vmax=15, cmap='bwr')
        ax3.set_title("Difference", c="navy", fontweight = 'bold')

        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label(r'$(\mathrm{MC} - \mathrm{ParaFlow})/\mathrm{MC}$ [%]')

        ax3.set_xticks([])
        ax3.set_yticks([])

        
        fig.savefig(self.result_path + "Average_Images.pdf", bbox_inches='tight')



    def plot_histograms(self):
        
        ''' Function for plotting one- and two-dimensional histograms of shower variables. '''


        ###### 1. histogram of brightest pixel 

        # get max values from the samples and GEANT4 data
        brightest_voxel_samples = np.max(self.samples, axis=(1,2)).squeeze()
        brightest_voxel_data    = np.max(self.data, axis=(1,2)).squeeze()

        # plot using the functions in plotting.py
        plot_histogram(brightest_voxel_samples/1000, brightest_voxel_data/1000, bin_centers = np.arange(0,78,1), xlim = (-1,79.5), plot_color = self.plot_color, error_color = self.error_color, title = 'Brightest~Voxel', x_label = 'Energy [GeV]', y_label = 'Events', filename = 'histogram_brightest_voxel', result_path = self.result_path, data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(brightest_voxel_data/1000, brightest_voxel_samples/1000, self.conditions, self.sample_conditions, title='Brightest~Voxel', colorbar_label='Energy [GeV]', result_path=self.result_path, bins = 10, file_name='2d_plot_brightest_voxel')



        ###### 2. histogram for sparsity  

        # get number of non-zero pixels
        num_zero_samples    = np.sum(self.samples > 0.0, axis=(1,2,3))
        num_zero_data       = np.sum(self.data > 0.0, axis=(1,2))

        # calculate sparsity
        sparsity_samples     = num_zero_samples / (self.image_size_x * self.image_size_y)
        sparsity_data        = num_zero_data / (self.image_size_x * self.image_size_y)

        # plot using the functions in plotting.py
        plot_histogram(sparsity_samples, sparsity_data, bin_centers = np.arange(0.58,1.02,0.03), xlim = (0.56, 1.04), plot_color = self.plot_color, error_color = self.error_color, title = 'Sparsity', x_label = 'Sparsity', y_label = 'Events', filename = 'histogram_sparsity', result_path = self.result_path, data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)




        ####### 3.  histogram of second brightest pixel 

        # flatten array to get second brightest pixel
        samples_flat = self.samples.reshape(self.sample_size, -1) # flatten first
        data_flat    = self.data.reshape(self.sample_size, -1)

        # Use np.partition to find the second largest element
        second_brightest_voxel_samples  = np.partition(samples_flat, -2, axis=1)[:, -2]
        second_brightest_voxel_data     = np.partition(data_flat, -2, axis=1)[:, -2]  
        
        # plot using the function in plotting.py
        plot_histogram(second_brightest_voxel_samples/1000, second_brightest_voxel_data/1000, bin_centers = np.arange(0,41.5,0.5), xlim = (-1, 43), plot_color = self.plot_color, error_color = self.error_color, title = '2nd~Brightest~Voxel', x_label = 'Energy [GeV]', y_label = 'Events', filename = 'histogram_second_brightest_voxel', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(second_brightest_voxel_data/1000, second_brightest_voxel_samples/1000, self.conditions, self.sample_conditions, title='2nd~Brightest~Voxel', colorbar_label='Energy [GeV]', result_path=self.result_path, bins = 10, file_name='2d_plot_second_brightest_voxel')



        #######  4. histogram for ratio            

        # compute the ratio of the brightest and second brightest voxel
        ratio_samples   = (brightest_voxel_samples - second_brightest_voxel_samples)/(brightest_voxel_samples + second_brightest_voxel_samples)
        ratio_data      = (brightest_voxel_data - second_brightest_voxel_data)/(brightest_voxel_data + second_brightest_voxel_data)

        # plot using the function in plotting.py
        plot_histogram(ratio_samples, ratio_data, bin_centers = np.arange(0,1.025,0.025), xlim = (-0.0625, 1), plot_color = self.plot_color, error_color = self.error_color, title = 'Ratio', x_label = 'Ratio', y_label = 'Events', filename = 'histogram_ratio_voxels', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(ratio_data, ratio_samples, self.conditions, self.sample_conditions, title='Ratio', colorbar_label='Ratio', result_path=self.result_path, bins = 10, file_name='2d_plot_ratio')




        #######  5. histogram for shower width 

        # calculate the shower shapes with the helper function in the beginning
        shower_width_samples    = shower_width(self.samples.squeeze())
        shower_width_data       = shower_width(self.data.squeeze())


        # plot using the function in plotting.py
        plot_histogram(shower_width_samples, shower_width_data, bin_centers = np.arange(0.25, 2.75, 2.25 / 30),  plot_color = self.plot_color, error_color = self.error_color, title = 'Shower~Width', x_label = 'Shower Width [Crystal Width]', y_label = 'Events', filename = 'histogram_shower_width', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(shower_width_data, shower_width_samples, self.conditions, self.sample_conditions, title='Shower~Width', colorbar_label='Shower Width [Crystal Width]', result_path=self.result_path, bins = 10, file_name='2d_plot_shower_width')




        ###### 6. histogram for mean difference of neighboring pixels (horizontal) 

        diff_h_samples = np.abs(np.diff(self.samples, axis = 2))
        diff_h_mean_samples = np.mean(diff_h_samples, axis = (1,2,3))

        diff_h_data = np.abs(np.diff(self.data, axis = 2))
        diff_h_mean_data = np.mean(diff_h_data, axis = (1,2))

        plot_histogram(diff_h_mean_samples, diff_h_mean_data, bin_centers = np.arange(0,1570,30), xlim = (-10,1600),  plot_color = self.plot_color, error_color = self.error_color, title = 'Diff.~Neigboring~Pixels~(H)', x_label = 'Difference [MeV]', y_label = 'Events', filename = 'histogram_difference_horizontal', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(diff_h_mean_data, diff_h_mean_samples, self.conditions, self.sample_conditions, title='Difference~Neigbors~(H)', colorbar_label='Energy Difference [MeV]', result_path=self.result_path, bins = 10, file_name='2d_plot_difference_neighbors_horizontal')





        ###### 7. histogram for mean difference of neighboring pixels (vertical) 

        diff_v_samples = np.abs(np.diff(self.samples, axis = 1))
        diff_v_mean_samples = np.mean(diff_v_samples, axis = (1,2,3))

        diff_v_data = np.abs(np.diff(self.data, axis = 1))
        diff_v_mean_data = np.mean(diff_v_data, axis = (1,2))


        plot_histogram(diff_v_mean_samples, diff_v_mean_data, bin_centers = np.arange(0,1570,30),  plot_color = self.plot_color, error_color = self.error_color, title = 'Diff.~Neigboring~Pixels~(V)', x_label = 'Difference [MeV]', y_label = 'Events', filename = 'histogram_difference_vertical', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(diff_v_mean_data, diff_v_mean_samples, self.conditions, self.sample_conditions, title='Difference~Neigbors~(V)', colorbar_label='Energy Difference [MeV]', result_path=self.result_path, bins = 10, file_name='2d_plot_difference_neighbors_vertical')




        ###### 8.  histogram of variance per image  

        variance_per_image_samples = np.var(self.samples/1000, axis = (1,2,3), ddof=1)
        variance_per_image_data = np.var(self.data/1000, axis = (1,2), ddof=1)

        plot_histogram(variance_per_image_samples, variance_per_image_data, bin_centers = np.arange(0,30,0.6),  plot_color = self.plot_color, error_color = self.error_color, title = 'Variance', x_label = 'Variance [GeV]', y_label = 'Events', filename = 'histogram_variance', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(variance_per_image_data, variance_per_image_samples, self.conditions, self.sample_conditions, title='Variance', colorbar_label='Variance [GeV]', result_path=self.result_path, bins = 10, file_name='2d_plot_variance')




        ###### 9. histogram of R9 

        r9_samples = calculate_r9(self.samples, image_size_x=self.image_size_x, image_size_y = self.image_size_y).squeeze()
        r9_data = calculate_r9(self.data, image_size_x=self.image_size_x, image_size_y = self.image_size_y).squeeze()

        plot_histogram(r9_samples, r9_data, bin_centers = np.arange(0.40, 1.025, 0.025), xlim =(0.375, 1.05),  plot_color = self.plot_color, error_color = self.error_color, title = 'R_9', x_label = r'$R_9$', y_label = 'Events', filename = 'histogram_r9', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(r9_data, r9_samples, self.conditions, self.sample_conditions, title='R_9', colorbar_label=r'$R_9$', result_path=self.result_path, bins = 10, file_name='2d_plot_r9')




        ###### 10. Distance barycenter - center

        barycenter_x_samples, barycenter_y_samples  = get_barycenter(self.samples)
        barycenter_x_data, barycenter_y_data        = get_barycenter(self.data)

        center_x, center_y = (self.image_size_y + 1) / 2, (self.image_size_x + 1) / 2

        distance_center_barycenter_samples  = np.sqrt( (barycenter_x_samples - center_x)**2 + (barycenter_y_samples - center_y)**2 )
        distance_center_barycenter_data     = np.sqrt( (barycenter_x_data - center_x)**2 + (barycenter_y_data - center_y)**2 )

        plot_histogram(distance_center_barycenter_samples, distance_center_barycenter_data, bin_centers = np.arange(0, 6.12, 0.12),  plot_color = self.plot_color, error_color = self.error_color, title = '|Barycenter~-~Center|', x_label = 'Distance [Crystal Width]', y_label = 'Events', filename = 'histogram_distance_center_barycenter', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(distance_center_barycenter_data, distance_center_barycenter_samples, self.conditions, self.sample_conditions, title='|Barycenter~-~Center|', colorbar_label='Distance [Crystal Width]', result_path=self.result_path, bins = 10, file_name='2d_plot_distance_center_barycenter')




        ###### 11. Distance barycenter - maximum

        max_x_samples, max_y_samples = np.unravel_index(np.argmax(self.samples.reshape(-1, self.image_size_x*self.image_size_y), axis = 1), (self.image_size_x, self.image_size_y))
        max_x_data, max_y_data       = np.unravel_index(np.argmax(self.data.reshape(-1, self.image_size_x*self.image_size_y), axis = 1), (self.image_size_x, self.image_size_y))

        distance_max_barycenter_samples = np.sqrt( (barycenter_x_samples - max_x_samples)**2 + (barycenter_y_samples - max_y_samples)**2 )
        distance_max_barycenter_data    = np.sqrt( (barycenter_x_data - max_x_data)**2 + (barycenter_y_data - max_y_data)**2 )

        plot_histogram(distance_max_barycenter_samples, distance_max_barycenter_data, bin_centers = np.arange(0, 15.375, 0.375),  plot_color = self.plot_color, error_color = self.error_color, title = '|Barycenter~-~Maximum|', x_label = 'Distance [Crystal Width]', y_label = 'Events', filename = 'histogram_distance_max_barycenter', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(distance_max_barycenter_data, distance_max_barycenter_samples, self.conditions, self.sample_conditions, title='|Barycenter~-~Maximum|', colorbar_label='Distance [Crystal Width]', result_path=self.result_path, bins = 10, file_name='2d_plot_distance_max_barycenter')



        
        ###### 12. Distance of Brightest and Third Brightest 

        distances_samples = distance_brigtest_x_brightest(self.samples, x = 3, image_size_x=self.image_size_x, image_size_y=self.image_size_y)
        distances_data    = distance_brigtest_x_brightest(self.data, x = 3, image_size_x=self.image_size_x, image_size_y=self.image_size_y)

        plot_histogram(distances_samples, distances_data, bin_centers = np.arange(0, 10, 1),  plot_color = self.plot_color, error_color = self.error_color, title = 'Distance~1st~&~3rd~Brightest', x_label = 'Distance [Crystal Width]', y_label = 'Events', filename = 'histogram_distance_third_brightest', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(distances_data, distances_samples, self.conditions, self.sample_conditions, title='Distance~1st~&~3rd~Brightest', colorbar_label='Distance [Crystal Width]', result_path=self.result_path, bins = 10, file_name='2d_plot_third_distance')




        ###### 13. Distance of Brightest and Fourth Brightest 

        distances_samples = distance_brigtest_x_brightest(self.samples, x = 4, image_size_x=self.image_size_x, image_size_y=self.image_size_y)
        distances_data    = distance_brigtest_x_brightest(self.data, x = 4, image_size_x=self.image_size_x, image_size_y=self.image_size_y)

        plot_histogram(distances_samples, distances_data, bin_centers = np.arange(1, 8, 1),  plot_color = self.plot_color, error_color = self.error_color, title = 'Distance~1st~&~4th~Brightest', xlim = (0.00,7.7), x_label = 'Distance [Crystal Width]', y_label = 'Events', filename = 'histogram_distance_fourth_brightest', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(distances_data, distances_samples, self.conditions, self.sample_conditions, title='Distance~1st~&~4th~Brightest', colorbar_label='Distance [Crystal Width]', result_path=self.result_path, bins = 10, file_name='2d_plot_fourth_distance')




        ###### 14. Sigma ieta ieta 


        sigma_ieta_ieta_value_samples = sigma_ieta_ieta(self.samples)
        sigma_ieta_ieta_value_data    = sigma_ieta_ieta(self.data)

        plot_histogram(sigma_ieta_ieta_value_samples, sigma_ieta_ieta_value_data, bin_centers = np.arange(0.00002, 0.0009,0.000015),  plot_color = self.plot_color, error_color = self.error_color, title = '\sigma_{i \eta i \eta}', x_label = '\sigma_{i \eta i \eta} [Crystal Width]', y_label = 'Events', filename = 'histogram_sigma_ietaieta', result_path = self.result_path,  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(sigma_ieta_ieta_value_data, sigma_ieta_ieta_value_samples, self.conditions, self.sample_conditions, title='\sigma_{i \eta i \eta}', colorbar_label=r'$\sigma_{i \eta i \eta}$ [Crystal Width]', result_path=self.result_path, bins = 10, file_name='2d_plot_sigma_ieta_ieta')


        plt.close('all')



    def cluster_shower_images(self):

        ''' Function for running the clustering algorithm DBSCAN on the dataset, see clustering.py '''

        from clustering import clustering_batch

        num_clusters_data = clustering_batch(self.data, eps = 1.5, min_samples=1.0, threshold=400)
        num_clusters_samples = clustering_batch(self.samples, eps = 1.5, min_samples=1.0, threshold=400)

        plot_histogram(num_clusters_samples, num_clusters_data, bin_centers = np.arange(1, 6, 1),  plot_color = self.plot_color, error_color = self.error_color, title = 'Counted~Cluster(s)', x_label = 'Number of DBSCAN clusters', y_label = 'Events', filename = 'histogram_clustering', result_path = self.result_path, xlim =(0.00,5.70),  data_conditions = self.conditions, sample_conditions = self.sample_conditions, thickness_range = self.thickness_range, bin_comparison_thickness = self.bin_comparison_thickness, distance_range = self.distance_range, bin_comparison_distance = self.bin_comparison_distance)
        if (self.thickness_bin is None and self.distance_bin is None): plot_2d_parameterspace(num_clusters_data, num_clusters_samples, self.conditions, self.sample_conditions, title='Counted~Cluster(s)', colorbar_label='Counted Cluster(s)', result_path=self.result_path, bins = 10, file_name='2d_plot_num_clusters')

    


    def nearest_neighbor_images(self):

        ''' Function for plotting samples and their nearest neighbors in the GEANT4 dataset. '''

        ############# CONDITIONS #############

        distance = torch.Tensor([50, 70, 90])
        thickness = torch.Tensor([0.5, 1.0, 1.5])
        energy = torch.Tensor([20e3, 30e3, 50e3, 75e3, 100e3]) # one image per energy

        # rescale and rearrange conditions
        conditions_per_image = torch.cartesian_prod(condition_scaling(thickness, lower_bound=0.5, higher_bound=1.5, use_torch=True), condition_scaling(distance, lower_bound=50, higher_bound=90, use_torch=True)).unsqueeze(0)
        energy_condition = condition_scaling(energy/1000, lower_bound = 20, higher_bound = 100, use_torch=True)[:, None].repeat(1, conditions_per_image.size(1)).unsqueeze(2)
        conditions_without_energy = conditions_per_image.repeat(energy.size(0),1,1)
        conditions = torch.cat((energy_condition, conditions_without_energy), dim = 2)

        # Sample images
        samples      = self.model(conditions.to(self.device)).sample(())

        # Iterate over energies, creating one image for each energy value
        for idx_energy, energy_sample in enumerate(samples):


            # postprocess the samples
            sample = energy_sample.reshape(9,-1)
            sample = postprocessing(sample=sample, E_inc = energy[idx_energy].to(self.device), sample_size=9, image_size_x = self.image_size_x, image_size_y=self.image_size_y, threshold= 1.1 * self.noise_level, alpha=self.alpha)

            # back on cpu and change type to float
            sample = sample.cpu().to(dtype=torch.float64).reshape(9, self.image_size_x, self.image_size_y,1)


            # Plotting ....
            fig, big_axes = plt.subplots(nrows=3, ncols=1, figsize=(8,20))


            for row, big_ax in enumerate(big_axes):
                big_ax.set_ylabel(f"d = {thickness[row].item():.2f} $X_0$", fontweight = "bold", color = "Navy", labelpad = 30, loc = 'center')
                big_ax.tick_params(labelcolor=(1, 1, 1, 0), top = False, bottom = False, left = False, right = False)
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                
                big_ax._frameon = False


            # indices in subplot grid for images
            subplot_idx = [1,2,3,7,8,9,13,14,15]
            for i in range(1,10):
                ax = fig.add_subplot(6,3, subplot_idx[i-1])

                ### Find nearest neigbor in Geant4:

                image = sample[i-1]
                epsilon = 1e-8
                flattened_image = np.log(image.flatten().numpy() + epsilon)
                data_flattened = np.log(self.data.reshape(-1, self.image_size_x * self.image_size_y) + epsilon)

                # Compute distances (Euclidean)
                distances = np.linalg.norm(data_flattened - flattened_image, axis=1)  # Shape: (N,)

                # Find the index of the minimum distance
                nearest_neighbor_idx = np.argmin(distances)

                # Retrieve the nearest neighbor image
                nearest_neighbor_image = self.data[nearest_neighbor_idx]




                if i <=3: ax.set_title(f"b = {distance[(i-1) % 5]} cm", fontsize = 25, c ='navy', fontweight = 'bold', pad=10)

                cmap = copy.copy(matplotlib.colormaps["viridis"])
                cmap.set_under("w")

                # Use LogNorm to apply logarithmic scaling to color mapping
                norm = mcolors.LogNorm(vmin=1, vmax=100e3)

                im = ax.imshow(image, cmap=cmap, norm=norm)

                if ((i-1) % 3 == 0): ax.set_ylabel("ParaFlow", loc = 'center')

                # Remove ticks and tick labels from the main image plot
                ax.set_xticks([])
                ax.set_yticks([])


                ax = fig.add_subplot(6,3, subplot_idx[i-1] + 3)
                nn_image = ax.imshow(nearest_neighbor_image, cmap=cmap, norm=norm)

                if ((i-1) % 3 == 0): ax.set_ylabel("Geant4", loc = 'center')

                
            cbar_ax = fig.add_axes([0.92, 0.12, 0.05, 0.75])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Energy [MeV]")


            fig.savefig(self.result_path + f"nearest_neighbors_{int((energy[idx_energy]/1000))}.pdf", bbox_inches='tight')



    def comparison_images(self):

        ''' Function for plotting samples and a random image in the GEANT4 dataset with the same conditions '''


        ############# CONDITIONS #############

        # energies for which to sample
        distance = torch.Tensor([50, 70, 90])
        thickness = torch.Tensor([0.5, 1.0, 1.5])
        energy = torch.Tensor([20e3, 30e3, 50e3, 75e3, 100e3])

        # rearange and rescale conditions
        conditions_per_image = torch.cartesian_prod(condition_scaling(thickness, lower_bound=0.5, higher_bound=1.5, use_torch=True), condition_scaling(distance, lower_bound=50, higher_bound=90, use_torch=True)).unsqueeze(0)
        conditions_per_image_unscaled = torch.cartesian_prod(thickness, distance).unsqueeze(0)
        energy_condition = condition_scaling(energy/1000, lower_bound = 20, higher_bound = 100, use_torch=True)[:, None].repeat(1, conditions_per_image.size(1)).unsqueeze(2)
        conditions_without_energy = conditions_per_image.repeat(energy.size(0),1,1)
        conditions_without_energy_unscaled = conditions_per_image_unscaled.repeat(energy.size(0),1,1)
        conditions = torch.cat((energy_condition, conditions_without_energy), dim = 2)
        conditions_unscaled = torch.cat((energy[:, None].repeat(1, conditions_per_image.size(1)).unsqueeze(2), conditions_without_energy_unscaled), dim = 2)


        # Sample images using the flow
        samples      = self.model(conditions.to(self.device)).sample(())


        for idx_energy, energy_sample in enumerate(samples):


            # postprocess the samples
            sample = energy_sample.reshape(9,-1)
            sample = postprocessing(sample=sample, E_inc = energy[idx_energy].to(self.device), sample_size=9, image_size_x = self.image_size_x, image_size_y=self.image_size_y, threshold= 1.1 * self.noise_level, alpha=self.alpha)

            # back on cpu and change type to float
            sample = sample.cpu().to(dtype=torch.float64).reshape(9, self.image_size_x, self.image_size_y,1)

            # Plotting ....
            fig, big_axes = plt.subplots(nrows=3, ncols=1, figsize=(8,20))


            for row, big_ax in enumerate(big_axes):
                big_ax.set_ylabel(f"d = {thickness[row].item():.2f} $X_0$", fontweight = "bold", color = "Navy", labelpad = 30, loc = 'center')
                big_ax.tick_params(labelcolor=(1, 1, 1, 0), top = False, bottom = False, left = False, right = False)
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                
                big_ax._frameon = False

            subplot_idx = [1,2,3,7,8,9,13,14,15]
            for i in range(1,10):
                ax = fig.add_subplot(6,3, subplot_idx[i-1])

                ### Get Geant4 sample with same conditions
                image = sample[i-1]

                # tolerance values for the parameters
                epsilon_energy = 2000
                epsilon_thickness = 0.05
                epsilon_distance = 2

                # masks for parameter values
                mask_energy = ((self.conditions[:,0] > (conditions_unscaled[idx_energy,i-1,0].item() - epsilon_energy)) & (self.conditions[:,0] < (conditions_unscaled[idx_energy,i-1,0].item() + epsilon_energy)))
                mask_thickness = ((self.conditions[:,1] > (conditions_unscaled[idx_energy,i-1,1].item() - epsilon_thickness)) & (self.conditions[:,1] < (conditions_unscaled[idx_energy,i-1,1].item() + epsilon_thickness)))
                mask_distance = ((self.conditions[:,2] > (conditions_unscaled[idx_energy,i-1,2].item() - epsilon_distance)) & (self.conditions[:,2] < (conditions_unscaled[idx_energy,i-1,2].item() + epsilon_distance)))

                mask = np.logical_and(np.logical_and(mask_energy, mask_thickness), mask_distance)

                # get images which fulfill the requirements
                masked_geant4 = self.data[mask]
                random_geant4_image = masked_geant4[np.random.randint(0,len(masked_geant4))]


                if i <=3: ax.set_title(f"b = {distance[(i-1) % 5]} cm", fontsize = 25, c ='navy', fontweight = 'bold', pad=10)

                cmap = copy.copy(matplotlib.colormaps["viridis"])
                cmap.set_under("w")

                # Use LogNorm to apply logarithmic scaling to color mapping
                norm = mcolors.LogNorm(vmin=1, vmax=100e3)

                im = ax.imshow(image, cmap=cmap, norm=norm)

                if ((i-1) % 3 == 0): ax.set_ylabel("ParaFlow", loc = 'center')

                # Remove ticks and tick labels from the main image plot
                ax.set_xticks([])
                ax.set_yticks([])


                ax = fig.add_subplot(6,3, subplot_idx[i-1] + 3)
                geant4_im = ax.imshow(random_geant4_image, cmap=cmap, norm=norm)

                ax.set_xticks([])
                ax.set_yticks([])

                if ((i-1) % 3 == 0): ax.set_ylabel("Geant4", loc = 'center')

                
            # set color bar
            cbar_ax = fig.add_axes([0.92, 0.12, 0.05, 0.75])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Energy [MeV]")


            fig.savefig(self.result_path + f"example_images_{int((energy[idx_energy]/1000))}.pdf", bbox_inches='tight')

        
    def show_correlation_matrix(self):

        ''' Function for computing the correlation between five brightest pixels and displaying as heatmap. '''

        # flatten for partitioning
        flattened_samples = self.samples.reshape(self.sample_size, -1)
        flattened_data = self.data.reshape(self.sample_size, -1)

        # get five brightest in descending order
        five_brightest_values_samples = np.partition(flattened_samples, -5, axis=1)[:, -5:]
        five_brightest_values_samples_sorted = np.sort(five_brightest_values_samples, axis = 1)[:,::-1]

        five_brightest_values_data = np.partition(flattened_data, -5, axis=1)[:, -5:]
        five_brightest_values_data_sorted = np.sort(five_brightest_values_data, axis = 1)[:,::-1]


        # Calculate the correlation matrix of the five brightest values across all images
        correlation_matrix_samples = np.corrcoef(five_brightest_values_samples_sorted, rowvar=False, ddof=1)
        correlation_matrix_data = np.corrcoef(five_brightest_values_data_sorted, rowvar=False, ddof = 1)


        # Now plotting ...
        fig, big_ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

        big_ax.tick_params(labelcolor=(1, 1, 1, 0), top = False, bottom = False, left = False, right = False)
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        
        big_ax._frameon = False

        title = "Correlation~of~Five~Brightest~Pixels"


        big_ax.set_title("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$", c="black", fontsize = 40, loc = 'left', pad = 25)


        # Plot the correlation matrix as a heatmap for samples
        ax = fig.add_subplot(1,2,1)
        ax.set_title("ParaFlow", c="navy", fontweight = 'bold')
        im = ax.imshow(correlation_matrix_samples, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)

        # set ticks for x and y axis
        labels=["1st", "2nd", "3rd", "4th", "5th"]

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)


        # Add correlation values as text in each cell
        for i in range(correlation_matrix_samples.shape[0]):
            for j in range(correlation_matrix_samples.shape[1]):
                text = f"{correlation_matrix_samples[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="black")


        # Repeat now for data...

        ax = fig.add_subplot(1,2,2)
        ax.set_title("MC", c="navy", fontweight = 'bold')
        im = ax.imshow(correlation_matrix_data, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)

        labels=["1st", "2nd", "3rd", "4th", "5th"]

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Add values as text in each cell
        for i in range(correlation_matrix_data.shape[0]):
            for j in range(correlation_matrix_data.shape[1]):
                text = f"{correlation_matrix_data[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="black")
        
    
        fig.savefig(self.result_path + "correlation_matrices.pdf", bbox_inches='tight')




    def train_binary_classifier(self, classifier_type):

        ''' 
        Here we train the classifier as a metric for the model. The relevant functions can be found in the classifier.py file.
        However, we utilize this Judge framework since it has a working sampling process which is needed for the training of the classifier.
        '''

        # make directory for classifier results

        os.makedirs(self.result_path + "classifier/" + classifier_type + "/models/", exist_ok=True)
        results_path = self.result_path + "classifier/" + classifier_type + "/"

        # hyperparameters 

        max_epoch = 999
        batch_size = 1000
        initial_lr = 1e-4
        num_conditions = 3
        noise = False
        normalize = True
        logit = False

        ### Instantiate the classifier ###

        classifier = ElectromagneticShowerClassifier(num_classes=1)
        classifier.train()
        classifier = classifier.to(self.device)


        # get data
        dataloader_train, dataloader_test, dataloader_validation = get_data(samples = self.samples, geant4 = self.data, sample_conditions = self.sample_conditions, geant4_conditions = self.conditions, num_conditions = num_conditions, image_size_x=self.image_size_x, image_size_y=self.image_size_y, batch_size = batch_size, normalize=normalize, logit = logit)

        # train classifier
        training_losses, validation_losses, learning_rates, last_epoch = train_classifier(max_epoch=max_epoch, initial_lr=initial_lr, classifier=classifier, dataloader_train=dataloader_train, dataloader_validation = dataloader_validation, device = self.device, results_path = results_path, classifier_type=classifier_type)
        
        # plot histories
        plot_history(training_losses=training_losses, validation_losses=validation_losses, last_epoch=last_epoch, results_path=results_path, normalize=normalize, classifier_type=classifier_type)
        plot_learning_rates(learning_rates = learning_rates, last_epoch=last_epoch, results_path=results_path, normalize=normalize, classifier_type=classifier_type)

        # plot roc curve
        plot_roc(classifier=classifier, dataloader_test=dataloader_test, results_path=results_path, normalize=normalize, device=self.device, classifier_type=classifier_type)
        