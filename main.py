import os
import sys
import argparse
import math
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import torch
import yaml
import numpy as np
from .train import CaloFastSim
from .evaluation import Judge

# torch.cuda.empty_cache()
 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Fast calorimeter simulations using normalising flows")
    parser.add_argument('-m', '--mode', type=str, help = 'Specify mode of program, can be train, evaluate or classifier')
    parser.add_argument('-d', '--device', type = str, help = 'Specify the cuda device to run on.')
    parser.add_argument('-s', '--samplesize', type = int, help = 'Specify the desired sample size.')
    parser.add_argument('-r', '--run_id', type = str, help = 'Specify run_id')
    parser.add_argument('-c', '--continue_training', type =bool, help = 'Should we continue training an old model?')
    parser.add_argument('-n', '--detector_noise', type = bool, help = 'Specify if we want to add detector noise.')
    parser.add_argument('-nl', '--noise_level', type = float, help = 'Specify the level of noise for evaluation.')
    parser.add_argument('-t', '--type_classifier', type = str, help = 'Specify the type of the classifier, can be CNN or DNN.')
    parser.add_argument('-g', '--generate_samples', type = bool, help = 'Specify if samples shall be generated or loaded.')
    parser.add_argument('-sb', '--shielding_bin', type = int, help = 'Specify the values of shielding for which to evaluate.')
    parser.add_argument('-db', '--distance_bin', type = int, help = 'Specify the values of distance for which to evaluate.')
    parser.add_argument('-cbs', '--compare_bins_shielding', type = bool, help = 'Specify if we want comparison plots.')
    parser.add_argument('-cbd', '--compare_bins_distance', type = bool, help = 'Specify if we want comparison plots.')
    parser.add_argument('-cc', '--count_clusters', type = bool, help = "Flag for if clustering should be performed.")

    args = parser.parse_args()

    # run id
    parameters = {
        "run_id" : args.run_id,

        # training hyperparameters
        "batch_size" : 1000,
        "max_epoch" : 999,
        "initial_lr" : 1e-3,
        "cosine_annealing" : True,

        # model hyperparameters
        "n_features" : 24*24,
        "image_size_x" : 24,
        "image_size_y" : 24,
        "n_conditions" : 3,
        "n_bins" : 10,
        "n_transforms" : 10,
        "n_aux_layers" : 2,
        "n_aux_nodes" : 196,
        "random_perm" : True,
        "p_dropout"   : 0.00,
        "weights"     : False,

        # conditions
        "detector_noise" : bool(args.detector_noise),

        # preprocessing
        "noise" : True,
        "noise_level" : 20e-3,
        "alpha" : 1e-6,
        "particle_type" : 'photon'
    }

    # if we train with noise, we have one addtional condition
    if (parameters['detector_noise']): parameters['n_conditions'] += 1

    # get mode of programm and noise level from parser
    mode = args.mode
    detector_noise_level = args.noise_level
    detector_noise = args.detector_noise


    reload_model = args.continue_training

    # If we reload our model, reload our parameters from the corresponding config file (overwrite the ones specified above)
    if reload_model:
        with open('/net/data_cms3a-1/kann/fast_calo_flow/results_data_florian/' + args.run_id + '/config.yml', 'r') as file:
            parameters = yaml.safe_load(file)


    print(f"Run:     {parameters['run_id']}")

    # device
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Working on device: {device:>6}")

    # path of dataset
    data_path = "/net/data_cms3a-1/kann/fast_calo_flow/data/shielding_distance/"
    if parameters['detector_noise']: data_path = data_path + 'noise/'

    
    if ( mode == "train" ):

        # instantiate training class and run its methods (see train.py)
        framework = CaloFastSim(data_path=data_path, device=device, reload_model=reload_model, **parameters, parameters=parameters)
        framework.training()
        framework.plot_history()
        framework.plot_learning_rates()
        framework.evaluate_test_loss()
        framework.sample_data()


    elif ( mode == 'evaluate'):

        # instantiate evaluation class and run its methods (see evaluation.py)
        framework_evaluation = Judge(run_id = parameters["run_id"], sample_size = args.samplesize, data_path = data_path, device = device, parameters=parameters, detector_noise_level = detector_noise_level, shielding_bin = args.shielding_bin, distance_bin=args.distance_bin, bin_comparison_shielding = args.compare_bins_shielding, bin_comparison_distance=args.compare_bins_distance, generate=args.generate_samples)
        framework_evaluation.plot_averages()
        framework_evaluation.plot_histograms()
        if args.count_clusters: framework_evaluation.cluster_shower_images()
        framework_evaluation.show_correlation_matrix()
        if (not detector_noise_level):
            framework_evaluation.calculate_testloss()
            # framework_evaluation.sample_more_images()
            framework_evaluation.comparison_images()

    elif ( mode == 'classifier'):

        # instantiate evaluation class and run the classifier (see evaluation.py & classifier.py)
        framework_evaluation = Judge(run_id = parameters["run_id"], sample_size = args.samplesize, data_path = data_path, device = device, parameters=parameters, detector_noise_level = detector_noise_level, shielding_bin = args.shielding_bin, distance_bin=args.distance_bin, bin_comparison_shielding = args.compare_bins_shielding, bin_comparison_distance=args.compare_bins_distance, generate=args.generate_samples)
        framework_evaluation.train_binary_classifier(classifier_type=args.type_classifier)