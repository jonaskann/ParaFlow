''' 
This file contains all classes and function necessary to train and evaluate the
binary classifier which is used as a metric for our generative model.
The usage of these functions and classes is integrated in the 'evaluation.py' file, 
since the generation of images is written in this framework.
'''

# Import necessary libraries
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


# for the roc metrics
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

# Imports from other files
from utils import EarlyStopper
from data import condition_scaling


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1), use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.30) if use_dropout else nn.Identity()

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        if self.use_dropout:
            out = self.dropout(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.01)
        return out

class ElectromagneticShowerClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(ElectromagneticShowerClassifier, self).__init__()
        # Image feature extractor
        self.layer1 = ResidualBlock(1, 16, stride=1, use_dropout=True)
        self.layer2 = ResidualBlock(16, 32, stride=(1,1), use_dropout=True)
        self.layer3 = ResidualBlock(32, 64, stride=(2,1), use_dropout=True)
        self.layer4 = ResidualBlock(64, 128, stride=(1,1), use_dropout=True)
        self.layer5 = ResidualBlock(128, 128, stride=(2,2), use_dropout=True)
        self.layer6 = ResidualBlock(128, 256, stride=(1,1), use_dropout=True)
        self.layer7 = ResidualBlock(256, 256, stride=(2,2), use_dropout=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Condition feature extractor
        self.condition_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.01)
        )
        
        # Combined classifier
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, conditions):
        # Image feature extraction
        out_img = self.layer1(x)
        out_img = self.layer2(out_img)
        out_img = self.layer3(out_img)
        out_img = self.layer4(out_img)
        out_img = self.layer5(out_img)
        out_img = self.layer6(out_img)
        out_img = self.layer7(out_img)
        out_img = self.global_avg_pool(out_img)
        out_img = torch.flatten(out_img, 1)
        
        # Condition feature extraction
        out_conditions = self.condition_fc(conditions)
        
        # Concatenate image and condition features
        combined_features = torch.cat((out_img, out_conditions), dim=1)
        
        # Classification
        out = self.fc(combined_features)
        return out  # Binary classification





########################## Data architecture & Preprocessing #######################

class ClassifierData(torch.utils.data.Dataset):
    """ 
    Class for the classifier data, to be implemented with pytorchs DataLoader 
    Args:       particle - Define the particle to train the classifier on
                particle_idx - Index of the particle in the particle array
                samples - Generated samples
                geant4 - Data from the GEANT4 dataset
                samples_conditions - Conditions of the sampled images
                geant4_conditions - Conditions of the GEANT4 dataset
                transforms - The transformations on the dataset
                mode - Specify purpose of dataset (training, validation or testing)

    
    """

    def __init__(self, samples, geant4, samples_conditions, geant4_conditions, transforms, mode, image_size_x, image_size_y):

        assert mode in ("train", "test", "validation"), "Please chose mode to be 'train', 'test' or 'validation'."

        self.image_size_x = image_size_x
        self.image_size_y = image_size_y

        self.transforms             = transforms
        self.samples                = samples.squeeze()
        self.monte_carlo            = geant4

        self.samples_conditions     = samples_conditions
        self.monte_carlo_conditions = geant4_conditions


        ### produce labels
        self.labels_full = np.concatenate((np.full(shape = (len(self.samples),), fill_value = 1), np.full(shape = (len(self.monte_carlo),), fill_value = 0)), axis = 0)

        ### concatenate data & conditions
        self.data_full = np.concatenate((self.samples, self.monte_carlo), axis = 0)
        self.conditions_full = np.concatenate((self.samples_conditions, self.monte_carlo_conditions), axis = 0)

        ### permute data

        permutation = np.random.permutation(self.data_full.shape[0])
        self.data_perm = self.data_full[permutation]
        self.labels_perm = self.labels_full[permutation]
        self.conditions_perm = self.conditions_full[permutation]


        ### split data in training, test, validation

        if (mode == 'train'):

            ### 50 % of data goes into training
            self.data = self.data_perm[:int(0.6 * self.data_perm.shape[0])].copy()
            self.labels = self.labels_perm[:int(0.6 * self.data_perm.shape[0])].copy()
            self.conditions = self.conditions_perm[:int(0.6 * self.data_perm.shape[0])].copy()


        elif (mode == 'test'):

            ### 40 % of data goes into testing
            self.data = self.data_perm[int(0.6 * self.data_perm.shape[0]):int(0.8 * self.data_perm.shape[0])].copy()
            self.labels = self.labels_perm[int(0.6 * self.data_perm.shape[0]):int(0.8 * self.data_perm.shape[0])].copy()
            self.conditions = self.conditions_perm[int(0.6 * self.data_perm.shape[0]):int(0.8 * self.data_perm.shape[0])].copy()

        elif (mode == 'validation'):

            ### 10 % of data for validation
            self.data = self.data_perm[int(0.8 * self.data_perm.shape[0]):].copy()
            self.labels = self.labels_perm[int(0.8 * self.data_perm.shape[0]):].copy()
            self.conditions = self.conditions_perm[int(0.8 * self.data_perm.shape[0]):].copy()


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        sample_images      = self.data[idx]
        sample_labels      = self.labels[idx]
        sample_conditions  = self.conditions[idx]


        # get data in dictionaries (reshape images into 1 channel for convolutional network)
        sample = {'images' : sample_images.reshape(1, self.image_size_x, self.image_size_y), 'labels' : sample_labels, 'conditions' : sample_conditions}

        # preform transformations on the samples
        sample = self.transforms(sample)

        return sample
    
### Now some classes for preprocessing

class ToTensor(object):
    """ Convert the numpy arrays to torch tensors. """

    def __call__(self, sample):

        images, labels, conditions = sample['images'].copy(), sample['labels'].copy(), sample['conditions'].copy()

        mod_sample = {'images' : torch.from_numpy(images), 'labels' : torch.from_numpy(np.array([labels])), 'conditions' : torch.from_numpy(conditions)}

        return mod_sample
    
class ProcessConditions(object):
    """ Transfrom the conditions appropriatly... """
    
    def __call__(self, sample):

        conditions = sample['conditions'].copy()

        processed_condition_energy = np.array([condition_scaling(conditions[0], lower_bound=20, higher_bound=100, use_torch=False)])
        processed_condition_thickness = np.array([condition_scaling(conditions[1], lower_bound=0.5, higher_bound=1.5, use_torch=False)])
        processed_condition_distance = np.array([condition_scaling(conditions[2], lower_bound = 50, higher_bound=90, use_torch=False)])

        processed_condition = np.concatenate((processed_condition_energy, processed_condition_thickness, processed_condition_distance), axis = 0)

        sample['conditions'] = processed_condition

        return sample



class NormalizeImage(object):
    """ Convert the numpy arrays to torch tensors. """

    def __call__(self, sample):

        images = sample['images'].copy()

        # pos_images = images + 1000

        # normalize images based on sum
        norm_images = np.divide(images, np.sum(images, axis = (1,2)))


        # little check if we still have zero value pixels
        #if (np.sum(norm_images < 0) > 0): print("Alert, value(s):", norm_images[norm_images < 0])

        sample['images'] = norm_images

        return sample



class LogitTransformation(object):
    """ Class for performing Logit transformation as in CaloFlow [https://arxiv.org/abs/2106.05285]"""

    def __init__(self, alpha = 1e-4):

        self.alpha = alpha

    def __call__(self, sample):

        images = sample['images'].copy()

        # logit transformation of the images
        u = self.alpha + (1 - 2*self.alpha) * images
        u_logit = np.log(u / (1 - u))


        sample['images'] = u_logit

        return sample
    

def get_data(samples, geant4, sample_conditions, geant4_conditions, num_conditions, image_size_x, image_size_y, batch_size, normalize, logit = False):

    ''' Function for loading the datasets, performing the preprocessing and loading the data into the dataloaders. '''

    # different preprocessing steps for different settings
    if (normalize and logit):
        transformations = torchvision.transforms.Compose([ProcessConditions(), NormalizeImage(), LogitTransformation(alpha = 1e-4), ToTensor()])
    elif (normalize and not logit):
        transformations = torchvision.transforms.Compose([ProcessConditions(), NormalizeImage(), ToTensor()])
    else:
        transformations = torchvision.transforms.Compose([ProcessConditions(), ToTensor()])



    # load datasets
    dataset_train = ClassifierData(samples = samples, geant4 = geant4, samples_conditions = sample_conditions, geant4_conditions = geant4_conditions, transforms = transformations, mode = 'train', image_size_x=image_size_x, image_size_y=image_size_y)
    dataset_test = ClassifierData(samples = samples, geant4 = geant4, samples_conditions = sample_conditions, geant4_conditions = geant4_conditions, transforms = transformations, mode = 'test', image_size_x=image_size_x, image_size_y=image_size_y)
    dataset_validation = ClassifierData(samples = samples, geant4 = geant4, samples_conditions = sample_conditions, geant4_conditions = geant4_conditions, transforms = transformations, mode = 'validation', image_size_x=image_size_x, image_size_y=image_size_y)


    # load dataLoader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)


    return dataloader_train, dataloader_test, dataloader_validation


########################## Training of the classifier #########################

def train_classifier(max_epoch, initial_lr, classifier, dataloader_train, dataloader_validation, device, results_path, classifier_type):

    '''
    Function for training the classifier.
    Args:       max_epoch - Maximum number of epoch to train to
                initial_lr - Initial learning rate of Optimizer
                classifier - the Instance of the classifier
                dataloader_train - The dataloader of the train set
                dataloader_validation - The dataloader of the validation set
                device - The device on which we train the classifier
                results_path - the path to save the models
                classifier_type - Type of classifier, can be DNN or CNN.
    '''


    # check for anomalies
    torch.autograd.set_detect_anomaly(True)

    # history file  (containing training and validation losses)
    history_file = results_path + "history_" + classifier_type +  ".txt"
    file = open(history_file, "w+").close()
        

    # lists for the course of the loss
    training_losses = []
    validation_losses = []
    learning_rates = []

    # for early stopping
    min_validationloss = np.inf

    # set weight_decay

    weight_decay = 1e-2
    if (classifier_type == 'DNN'): weight_decay = 1e-6

    ### Setup Optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), initial_lr, weight_decay = weight_decay)

    # Scheduler for reducing learning rate on plateu
    scheduler_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 6)

    # Instantiate EarlyStopper
    early_stopper = EarlyStopper(patience = 20, min_delta=0.00)

    ##### Actual training loop #####
    for epoch in range(1, max_epoch +1):

        # Save lr for current epoch in corresponding array
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # set classifier to training loop (important for dropout)
        classifier = classifier.train()

        # variables for calculating the loss and accuracy
        running_loss = 0
        total = 0
        correct = 0

        # iterate over batches in dataloader
        for idx_batch, batch in enumerate(dataloader_train):

            # get images, conditions and labels
            images, labels, conditions  = batch['images'], batch['labels'], batch['conditions']

            # set datatype and device of each image and condition
            images = images.to(device).to(dtype = torch.float32)
            labels = labels.to(device).to(dtype = torch.float32)
            conditions = conditions.to(device).to(dtype = torch.float32)


            # get outputs of classifer
            outputs = classifier(images, conditions)

            # reset optimizer gradients
            optimizer.zero_grad()

            # calculate loss
            loss = criterion(outputs, labels)

            # calculate gradients based on loss
            loss.backward()

            # clip the gradients
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            # apply gradients to weights
            optimizer.step()

            # calculation of accuracy
            predicted = (torch.sigmoid(outputs.detach()) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # save training loss
            running_loss += loss.detach().item()

        # calculating training & validation loss & accuracy for each epoch
        with torch.no_grad():

            # calc accuracy
            epoch_accuracy = correct / total

            # set flow to evaluation mode (important for dropout)
            classifier = classifier.eval()

            # calculate training loss
            training_loss = running_loss / (idx_batch + 1)
            training_losses.append(training_loss)

            # now validation loss
            running_loss = 0

            for i, batch in enumerate(dataloader_validation):

                images, labels, conditions = batch['images'], batch['labels'], batch['conditions']
                    
                images = images.to(device).to(dtype = torch.float32)
                labels = labels.to(device).to(dtype = torch.float32)
                conditions = conditions.to(device).to(dtype = torch.float32)


                # calculate loss (take mean over batch)
                outputs = classifier(images, conditions)
                loss = criterion(outputs, labels)

                running_loss += loss.item()



            validation_loss = running_loss / (i + 1)
            validation_losses.append(validation_loss)

            # Print feedbacl
            print(f"\033[1;34mEpoch: {epoch}\033[0m, \033[1;32mTraining loss: {training_loss:.4f}\033[0m, \033[1;31mValidation Loss: {validation_loss:.4f}\033[0m, \033[1;32mAccuracy: {epoch_accuracy:.4f}\033[0m")


            # saving loss values in history.txt
            with open(history_file, "a+") as file:
                file.write(f"Epoch: {epoch:>3}, Training loss:      {training_loss:.4f}, Validation loss:     {validation_loss:.4f}, Accuracy:      {epoch_accuracy:.4f}\n")


            # Update learning rate based on validation loss (checked once an epoch)
            scheduler_plateu.step(validation_loss)


            # if the current epoch is the best epoch, save it as the new best model
            if (validation_loss < min_validationloss): 
                min_validationloss = validation_loss
                # leave the second best model as backup if something goes wrong in saving process... has happened before :/
                if os.path.isfile(results_path + "models/best_model.pt"): os.rename(results_path + "models/best_model.pt", results_path + "models/backup.pt") 
                
                # save not just model, but also history, optimizer, lr scheduler
                checkpoint = { 
                    'epoch': epoch,
                    'training_loss' : training_losses,
                    'validation_loss' : validation_losses,
                    'learning_rates' : learning_rates,
                    'model': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': scheduler_plateu.state_dict()}
                
                # Save everything in pt file
                torch.save(checkpoint, results_path + "models/best_model.pt")


            # Check for early stopping or if the maximum number of epochs has been reached
            if (early_stopper.early_stop(float(validation_loss)) or epoch == max_epoch):

                # get best epoch for printing purposes
                best_epoch = np.argmin(np.array( validation_losses ))+1
                print( 'Epoch with lowest loss: ', np.min(np.array( validation_losses )) , ' at epoch: ', best_epoch)
                
                # Reload the best model for further evaluation
                classifier.load_state_dict(torch.load(results_path + 'models/best_model.pt')['model'])
                classifier.eval()
                break

    return training_losses, validation_losses, learning_rates, epoch


#################### Plotting / Evaluation functions ####################

#### Plot the course of the losses ###
def plot_history(training_losses, validation_losses, last_epoch, results_path ,normalize, classifier_type):

    # array for epochs
    epochs_array = np.arange(1, last_epoch+1)

    # plotting history
    fig = plt.figure()

    plt.plot(epochs_array, training_losses, c = "blue", lw = 2, label = "Training loss", marker = "o", markersize = 5.0)
    plt.plot(epochs_array, validation_losses, c = "red", lw = 2, label = "Validation loss", marker = "x", markersize = 5.0)

    plt.xlabel("Epoch", fontsize = 30)
    plt.ylabel("Loss", fontsize = 30)
    plt.legend(fontsize = 30)
    plt.grid()

    if normalize:
        info = 'norm_'
    else:
        info=''

    fig.savefig(results_path + "history_" + str(info) + classifier_type + ".pdf", bbox_inches = 'tight')


#### Plot the course of the learning rate ###
def plot_learning_rates(learning_rates, last_epoch, results_path ,normalize, classifier_type):

    # array for epochs
    epochs_array = np.arange(1, last_epoch+1)

    # plotting history
    fig = plt.figure()

    plt.plot(epochs_array, learning_rates, c = "blue", lw = 2, label = "Learning rate", marker = "o", markersize = 5.0)

    plt.xlabel("Epoch", fontsize = 30)
    plt.ylabel("Learning Rate", fontsize = 30)
    plt.legend()
    plt.grid()

    if normalize:
        info = 'norm_'
    else:
        info=''

    fig.savefig(results_path + "learning_rate_" + str(info) + classifier_type + ".pdf", bbox_inches = 'tight')


#### Plot the ROC Curve ####
def plot_roc(classifier, dataloader_test, results_path, normalize, classifier_type, device):
        
    with torch.no_grad():

        # set classifier to evaluation (important for dropout)
        classifier = classifier.eval()

        # get batch (batch_size = dataset size)
        for batch in dataloader_test:
            test_data, test_labels, test_conditions = batch['images'], batch['labels'], batch['conditions']

        test_data = test_data.to(device).to(torch.float32)
        test_labels = test_labels.to(device).to(torch.float32)
        test_conditions = test_conditions.to(device).to(torch.float32)

        # get predictions
        predicted_logits = classifier(test_data, test_conditions)
        predicted_probabilities = torch.sigmoid(predicted_logits.squeeze())


        predicted_classes = (predicted_probabilities.cpu().numpy() > 0.5).astype(np.float32)
        correct  = np.sum(predicted_classes == test_labels.squeeze().cpu().numpy())
        accuracy = correct / len(test_labels)


    # change filenames depending on if normalization is on or not
    if normalize:
        info = 'norm_'
    else:
        info=''

    # Calculate the ROC curve points and AUC
    fpr, tpr, _ = roc_curve(test_labels.cpu().numpy(), predicted_probabilities.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    print(f"AUC:     {roc_auc:.4f}, Accuracy:       {accuracy:.4f}")

    with open(results_path + "results_" + str(info) + ".txt", "w+") as file:
        file.write(f"AUC:     {roc_auc:.4f}, Accuracy:       {accuracy:.4f}")
    
    # Plot the ROC curve
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.title('ROC Curve - CNN', fontsize = 20)
    plt.legend(loc="lower right", fontsize = 20)

    fig.tight_layout()
    fig.savefig(results_path + "roc_" + str(info) + classifier_type + ".pdf", bbox_inches = 'tight')

