'''
File with relevant classes and functions for the training procedure.
'''


### Libraries
import os
import shutil

import typing
import math
import copy
import yaml

import numpy as np

import zuko

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision


# Load the custom model
from custom_model.model import NSF

# Load data classes
from data import SimData, FlattenImage, NormalizeImage, AddNoise, ToTensor, LogitTransformation, postprocessing, condition_scaling, AddWeights
from utils import EarlyStopper


# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.CMS])


class CaloFastSim:
    '''
    Class for actual model, training & sampling.
    Args:   data_path - path to the dataset
            max_epoch - the maximum epoch to train
            initial_lr - the initial learning rate before using the learning rate scheduler
            particle_type - the type of particle to train on (pion or photon)
            batch_size - size of minibatches
            n_features - number of input dimensions
            n_conditions - number of conditions of the model
            n_bins - number of bins for spline transformations
            n_transforms - number of transformations in flow (number of MADE blocks)
            n_aux_layers - number of hidden layers in the MADE block
            n_aux_nodes - number of nodes in the hidden layer
            random_perm - boolean, if we permute between transformations
            p_dropout - float, percentage of dropout in the MADE block
            device - device to train on
            reload_model - if we continue training or start over
            run_id - name of the run
            cosine_annealing - boolean, if we use cosine annealing as lr scheduler
            noise - boolean, if noise is added in preprocessing
            noise_level - upper boundary of the noise added in preprocessing
            alpha - parameter for logit transformation in preprocessing

    '''

    def __init__(self, data_path : str, max_epoch: int, initial_lr : float, batch_size : int, n_features : int, image_size_x : int, image_size_y : int, n_conditions : int, n_bins : int, n_transforms : int, n_aux_layers : int, n_aux_nodes : int, random_perm : bool, p_dropout : float, device : str, reload_model : bool, run_id : str, noise : bool, cosine_annealing : bool, noise_level : float, alpha: float, weights : bool, parameters) -> None:

        # assign run_id to keep track of results (for config file)
        self.run_id = run_id

        # seed for random number generator
        self.torch_seed = 42

        # hyperparameters for training details
        self.max_epoch  = max_epoch
        self.batch_size = batch_size
        self.initial_lr = initial_lr

        # the device
        self.device = device

        # switch for retraining or continue of training
        self.reload_model = reload_model


        # hyperparameters of the flow
        self.n_features     = n_features
        self.image_size_x   = image_size_x
        self.image_size_y   = image_size_y
        self.n_conditions   = n_conditions
        self.n_bins         = n_bins
        self.n_transforms   = n_transforms
        self.n_aux_layers   = n_aux_layers
        self.n_aux_nodes    = n_aux_nodes
        self.random_perm    = random_perm
        self.p_dropout      = p_dropout
        self.noise = noise
        self.switch_cosineannealing = cosine_annealing
        self.weights = weights

        # for early stopping and model saving
        self.min_validationloss = np.inf

        # Preprocessing parameters
        self.noise_level = noise_level
        self.alpha = alpha

        # path to data & results
        self.data_path = data_path
        self.results_path = f"/net/data_cms3a-1/kann/fast_calo_flow/results/{run_id}/"

        # set up dump directories for results
        os.makedirs(self.results_path, exist_ok=reload_model) # abort when trying to override run that exists / if we reload the model its okay that the directory already exists

        # subdirectories in results directory
        os.makedirs(self.results_path + "images/", exist_ok=True)
        os.makedirs(self.results_path + "models/", exist_ok=True)
        os.makedirs(self.results_path + "evaluation/", exist_ok=True)


        # Save configuration parameters in YAML-file
        self.config_file = self.results_path + 'config.yml'

        if (not self.reload_model): ### If we activated reload model, parameters have already been dumped
            with open(self.config_file, 'a') as file:
                yaml.dump(parameters, file, default_flow_style=False)

        # For monitoring
        if (not self.reload_model):
            # if we don't reload a pretrained model, instantiate lists
            self.training_losses = []
            self.validation_losses = []
            self.learning_rates = []
        else:
            shutil.copy2(self.results_path + 'models/best_model.pt', self.results_path + 'models/best_model_last_run.pt') # copy best model from last training run as backup
            # otherwise load lists from previous run
            self.checkpoint = torch.load(self.results_path + 'models/best_model.pt')
            self.training_losses = self.checkpoint['training_loss']
            self.validation_losses = self.checkpoint['validation_loss']
            self.learning_rates = self.checkpoint['learning_rates']


        # History file  (containing training and validation losses)
        self.history_file = self.results_path + "history.txt"


    def get_data(self):

        """ Function for loading the datasets and preprocessing transformations. """
            
        # Random number generator
        generator = np.random.default_rng(self.torch_seed)

        # Preprocessing Transformations (defined in data.py)
        self.transforms = torchvision.transforms.Compose([
                                        AddNoise(active=self.noise, noise_level = self.noise_level, generator = generator), 
                                        FlattenImage(self.n_features), 
                                        NormalizeImage(), 
                                        LogitTransformation(alpha=self.alpha),
                                        AddWeights(active=self.weights), 
                                        ToTensor()])

        
        # Get datasets for training, validation and testing (and depending on particle type)
        self.dataset_train          = SimData(data_dir = self.data_path, mode = "train", transforms = self.transforms)
        self.dataset_validation     = SimData(data_dir = self.data_path, mode = "validation", transforms = self.transforms)
        self.dataset_test           = SimData(data_dir = self.data_path, mode = "test", transforms = self.transforms)


    def training(self):

        """ Function for training the model. """

        # Instantiate customized zuko flow
        flow = NSF(features = self.n_features, context = self.n_conditions, bins = self.n_bins, transforms = self.n_transforms, randperm = self.random_perm, hidden_features = [self.n_aux_nodes] * self.n_aux_layers, p_dropout = self.p_dropout)

        # If reload_model is true, load the pretrained weights from checkpoint
        if (self.reload_model): 
            flow.load_state_dict(self.checkpoint['model'])
            flow = flow.train() # set flow to training mode (only important if dropout activated)
        
        # Load the flow to specified device
        flow = flow.to(device=self.device) 


        # Get the number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, flow.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of Parameters:       {params:,}")



        # Instantiate Optimizer (ADAM) and reload the state if specified
        optimizer = torch.optim.Adam(flow.parameters(), self.initial_lr)
        if (self.reload_model): 
            optimizer.load_state_dict(self.checkpoint['optimizer'])

        # Optimizer & scheduler for reducing learning rate on plateau
        optimizer_plateau = torch.optim.Adam(flow.parameters(), self.initial_lr)
        if (self.reload_model):
            optimizer_plateau.load_state_dict(self.checkpoint['optimizer_plateau'])

        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_plateau, mode = 'min', factor = 0.5, patience = 10)         # args: 'min' specifies that we want to minimize the lr, factor by how much we decrease it, patience gives the number of epochs after which we decrease the lr
        if (self.reload_model): 
            scheduler_plateau.load_state_dict(self.checkpoint['lr_sched_plateau'])


        # Get preprocessed data from corresponding method (see above)
        self.get_data()


        # Dataloaders for training & validation
        self.dataloader_train        = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
        self.dataloader_validation   = torch.utils.data.DataLoader(self.dataset_validation, batch_size=self.batch_size, shuffle=True)


        # Instantiate EarlyStopper
        early_stopper = EarlyStopper(patience = 15, min_delta=0.00)


        # Cosine annealing learning rate scheduler (if specified)
        if self.switch_cosineannealing: 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(self.dataloader_train), eta_min = 0.0)         # args: T_max gives the maximum number of iterations (after that lr gets reseted to initial), here the batch size; eta_min gives the minimu lr
            if self.reload_model:
                scheduler.load_state_dict(self.checkpoint['lr_sched'])

        


        # Set starting epoch (maybe load from previous training)
        epoch = self.checkpoint['epoch'] + 1 if (self.reload_model) else 1


        # Trainings loop
        while epoch <= self.max_epoch:

            # set flow to training mode
            flow = flow.train()


            # iterate over batches in dataloader
            for _, batch in enumerate(self.dataloader_train):

                # get images, conditions and weights
                images, conditions, weights = batch['images'], batch['conditions'], batch['weights']


                # set datatype and device of each image and condition
                images = torch.squeeze(images).to(self.device).to(dtype = torch.float32)
                conditions = conditions.to(self.device).to(dtype = torch.float32)


                # reset optimizer gradients
                optimizer.zero_grad(set_to_none=True)

                # calculate loss (take mean over batch)
                loss = -flow(conditions).log_prob(images) * weights.to(self.device)
                loss = loss.mean()

                # calculate gradients based on loss
                loss.backward()

                # apply gradients to waits
                optimizer.step()

                # if cosine annealing is specified, update lr scheduler
                if self.switch_cosineannealing: scheduler.step()


            # calculating training & validation loss for each epoch
            with torch.no_grad():

                # set flow to evaluation mode (important for dropout)
                flow = flow.eval()

                # calculate the losses using the corresponding method (see below)
                training_loss, validation_loss = self.calculate_losses(flow = flow, epoch = epoch)

                # saving loss values in history.txt
                with open(self.history_file, "a+") as file:
                    file.write(f"Epoch: {epoch:>3}, Training loss:      {training_loss:.3f}, Validation loss:     {validation_loss:.3f}\n")


                # Update learning rate based on validation loss (checked once an epoch)
                scheduler_plateau.step( validation_loss )

                # set the new learning rate into optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = float(scheduler_plateau.optimizer.param_groups[0]['lr'])

                # Syncronize learning rates across optimizers -> give cosine annealing optimizer the new reduced lr
                if (self.switch_cosineannealing):
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= len(self.dataloader_train), eta_min = 0)

                # Save lr for current epoch in corresponding array
                self.learning_rates.append(scheduler.optimizer.param_groups[0]['lr'])

                # make flow attribute and therefore available for other methods
                self.flow = flow


                # if the current epoch is the best epoch, save it as the new best model
                if (validation_loss < self.min_validationloss): 
                    self.min_validationloss = validation_loss
                    # leave the second best model as backup if something goes wrong in saving process... 
                    if os.path.isfile(self.results_path + "models/best_model.pt"): os.rename(self.results_path + "models/best_model.pt", self.results_path + "models/backup.pt") 
                    
                    # save not just model, but also history, optimizer, lr scheduler
                    checkpoint = { 
                        'epoch': epoch,
                        'training_loss' : self.training_losses,
                        'validation_loss' : self.validation_losses,
                        'learning_rates' : self.learning_rates,
                        'model': flow.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_sched': scheduler.state_dict(),
                        'optimizer_plateau': optimizer_plateau.state_dict(),
                        'lr_sched_plateau': scheduler_plateau.state_dict()}
                    
                    # Save everything in pt file
                    torch.save(checkpoint, self.results_path + "models/best_model.pt")


                # Check for early stopping or if the maximum number of epochs has been reached
                if (early_stopper.early_stop(float(validation_loss)) or epoch == self.max_epoch):

                    # get best epoch for printing purposes
                    best_epoch = np.argmin(np.array( self.validation_losses ))+1
                    print( 'Epoch with lowest loss: ', np.min(np.array( self.validation_losses )) , ' at epoch: ', best_epoch)
                    
                    # Reload the best model for further evaluation
                    flow.load_state_dict(torch.load(self.results_path + 'models/best_model.pt')['model'])
                    flow.eval()
                    self.flow = flow
        
                    break


                # For every 20th epoch, plot some samples
                if (epoch % 20 == 0): self.sample_data()


                epoch += 1


    @torch.no_grad()
    def calculate_losses(self, flow, epoch):

        '''
        Function for calculating the validation & training losses after each epoch and storing them for later use.
        '''

        flow = flow.eval()

        # training loss
        cummulative_training_loss = 0

        for i, batch in enumerate(self.dataloader_train):

            images, conditions, weights = batch['images'], batch['conditions'], batch['weights']
                
            images = torch.squeeze(images).to(self.device).to(dtype = torch.float32)
            conditions = conditions.to(self.device).to(dtype = torch.float32)

            training_loss = -flow(conditions).log_prob(images) * weights.to(self.device)
            cummulative_training_loss += training_loss.mean()

        training_loss = cummulative_training_loss / (i+1)
        self.training_losses.append(training_loss.cpu())

                

        # validation loss
        cummulative_validation_loss = 0

        for i, batch in enumerate(self.dataloader_validation):

            images, conditions, weights = batch['images'], batch['conditions'], batch['weights']
                
            images = torch.squeeze(images).to(self.device).to(dtype = torch.float32)
            conditions = conditions.to(self.device).to(dtype = torch.float32)

            validation_loss = -flow(conditions).log_prob(images) * weights.to(self.device)
            cummulative_validation_loss += validation_loss.mean()

        validation_loss = cummulative_validation_loss / (i + 1)
        self.validation_losses.append(validation_loss.cpu())


        # Print feedback
        print(f"\033[1;34mEpoch: {epoch}\033[0m, \033[1;32mTraining loss: {training_loss:.3f}\033[0m, \033[1;31mValidation Loss: {validation_loss:.3f}\033[0m")

        return training_loss, validation_loss

    @torch.no_grad()
    def evaluate_test_loss(self) -> None:

        """
        Function for calculating the test loss at the end of training
        """

        # Instantiate dataloader
        dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test, batch_size=self.batch_size, shuffle=True)

        self.flow = self.flow.eval()

        # Calculate test loss:
        cummulative_test_loss = 0
        for i, batch in enumerate(dataloader_test):

            images, conditions, weights = batch['images'], batch['conditions'], batch['weights']


            images = torch.squeeze(images).to(self.device).to(dtype = torch.float32)
            conditions = conditions.to(self.device).to(dtype = torch.float32)


            loss = -self.flow(conditions).log_prob(images) * weights.to(self.device)
            cummulative_test_loss += loss.mean().cpu()

        test_loss = cummulative_test_loss / (i + 1)


        # Print feedback
        print("-"*70)
        print(f"\033[1;31mTest loss: {test_loss:.3f}\033[0m")



    @torch.no_grad()
    def plot_history(self) -> None:

        '''
        Function for plotting curves of the training and validation loss.
        '''

        # array for epochs
        epochs_array = np.arange(1, len(self.training_losses) + 1)

        # plotting history
        fig = plt.figure()

        plt.plot(epochs_array, self.training_losses, c = "blue", lw = 2, label = "Training loss", marker = "o", markersize = 5.0)
        plt.plot(epochs_array, self.validation_losses, c = "red", lw = 2, label = "Validation loss", marker = "x", markersize = 5.0)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        fig.savefig(self.results_path + "images/history.pdf", bbox_inches = 'tight')

    @torch.no_grad()
    def plot_learning_rates(self) -> None:
        '''
        Function for plotting the course of the learning rate over each epoch.
        '''

        # array for epochs
        batch_array = np.arange(1, len(self.learning_rates) + 1)


        # plotting history
        fig = plt.figure()

        plt.plot(batch_array, self.learning_rates, c = "blue", lw = 2, label = "Learning rate", marker = "o", markersize = 5.0)

        plt.xlabel("Batch")
        plt.ylabel("Learning rate")
        plt.legend()
        plt.grid()

        plt.show()
        fig.savefig(self.results_path + "images/history_lr.png", dpi = 350)


    def sample_data(self):
        '''
        Function for sampling shower images during training
        '''

        with torch.no_grad():

            self.flow = self.flow.eval()

            # Parameters for which to sample
            distance = torch.Tensor([50, 70, 90])
            thickness = torch.Tensor([0.5, 1.0, 1.5])
            energy = torch.Tensor([20e3, 50e3, 100e3])

            # Scale & Rearrange these conditions for sampling
            conditions_per_image = torch.cartesian_prod(condition_scaling(thickness, lower_bound= 0.5, higher_bound=1.5, use_torch=True), condition_scaling(distance, lower_bound=50, higher_bound=90, use_torch=True)).unsqueeze(0)
            energy_condition = condition_scaling(energy/1000, lower_bound = 20, higher_bound = 100, use_torch=True)[:, None].repeat(1, conditions_per_image.size(1)).unsqueeze(2)
            conditions_without_energy = conditions_per_image.repeat(energy.size(0),1,1)
            conditions = torch.cat((energy_condition, conditions_without_energy), dim = 2)

            # Sample images
            samples      = self.flow(conditions.to(self.device)).sample(())


            # Plot distinct grid of shower images for each energy
            for idx_energy, energy_sample in enumerate(samples):

                # Postprocess the samples
                sample = energy_sample.reshape(9,self.n_features)
                sample = postprocessing(sample=sample, E_inc = energy[idx_energy].to(self.device), sample_size=9, image_size_x = self.image_size_x, image_size_y=self.image_size_y, threshold= 10 * self.noise_level, alpha=self.alpha, noise = None)

                # Images on cpu and reshaped
                sample = sample.cpu().to(dtype=torch.float64).reshape(9, self.image_size_x, self.image_size_y,1)

                # Now plotting
                fig, big_axes = plt.subplots(nrows=3, ncols=1, figsize=(15,17))

                # Outer axes for different values of thickness
                for row, big_ax in enumerate(big_axes):
                    big_ax.set_title(f"Thickness {thickness[row].item():.1f} $X_0$", fontweight = "bold", color = "Navy", loc = "left", pad = 30)

                    big_ax.tick_params(labelcolor=(1, 1, 1, 0), top = False, bottom = False, left = False, right = False)
                    big_ax.set_xticks([])
                    big_ax.set_yticks([])
                    
                    big_ax._frameon = False

                for i in range(1,10):
                    ax = fig.add_subplot(3,3, i)

                    ax.set_title(f"Distance: {distance[(i-1) % 3]} cm", fontsize = 20, fontweight = 'bold', pad=10)

                    cmap = copy.copy(matplotlib.colormaps["viridis"])
                    cmap.set_under("w")

                    # Use LogNorm to apply logarithmic scaling to color mapping
                    norm = mcolors.LogNorm(vmin=1, vmax=100e3)
    
                    im = ax.imshow(sample[i-1], cmap=cmap, norm=norm)

                    # Remove ticks and tick labels from the main image plot
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Rechter Farbbalken für positive Werte
                    cbar = fig.colorbar(im, ax = ax)
                    cbar.set_label("Energy [MeV]")

                plt.close('all')
                fig.savefig(self.results_path + f"images/samples_end_{int((energy[idx_energy]/1000))}.pdf", bbox_inches='tight')







