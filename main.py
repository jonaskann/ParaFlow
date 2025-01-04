import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import torch
import yaml
import numpy as np
from train import CaloFastSim
from evaluation import Judge
 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Parametrized fast calorimeter simulations using normalising flows.")

    parser.add_argument('-m', '--mode', type=str, help = 'Specify mode of program, can be train, evaluate or classifier.')
    parser.add_argument('-d', '--device', type = str, help = 'Specify the cuda device to run on.')
    parser.add_argument('-s', '--samplesize', type = int, help = 'Specify the desired sample size for evaluation.')
    parser.add_argument('-r', '--run_id', type = str, help = 'Specify run_id for identification.')
    parser.add_argument('-c', '--continue_training', type =bool, help = 'Flag for continuing the training an exisiting model.')

    parser.add_argument('-g', '--generate_samples', type = bool, help = 'Flag if new samples shall be generated or existing ones loaded.')
    parser.add_argument('-tb', '--thickness_bin', type = int, help = 'Specify the range of thickness of the absorber material for which to evaluate.')
    parser.add_argument('-db', '--distance_bin', type = int, help = 'Specify the range of distance of the absorber material for which to evaluate.')
    parser.add_argument('-cbt', '--compare_bins_thickness', type = bool, help = 'Flag for comparison plots between different thickness ranges.')
    parser.add_argument('-cbd', '--compare_bins_distance', type = bool, help = 'Flag for comparison plots between different distance ranges.')
    parser.add_argument('-cc', '--count_clusters', type = bool, help = "Flag if clustering should be performed.")

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
        "n_features" : 16*8,
        "image_size_x" : 16,
        "image_size_y" : 8,
        "n_conditions" : 3,
        "n_bins" : 10,
        "n_transforms" : 10,
        "n_aux_layers" : 2,
        "n_aux_nodes" : 196,
        "random_perm" : True,
        "p_dropout"   : 0.00,
        "weights"     : False,

        # preprocessing
        "noise" : True,
        "noise_level" : 20e-3,
        "alpha" : 1e-6,
    }

    # get mode of programm from parser
    mode = args.mode
    assert mode in ('train', 'evaluate', 'classifier'), 'Mode must be train, evaluate or classifier.'
    reload_model = args.continue_training

    # If we reload our model, reload our parameters from the corresponding config yaml (overwrite the ones specified above)
    if reload_model:
        with open('/net/data_cms3a-1/kann/fast_calo_flow/results/' + args.run_id + '/config.yml', 'r') as file:
            parameters = yaml.safe_load(file)


    print(f"Run:     {parameters['run_id']}")


    # Specify working device
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Working on device: {device:>6}")

    # Path of dataset
    data_path = "/net/data_cms3a-1/kann/public/ParaFlow/data/"
    
    if ( mode == "train" ):

        # Instantiate training class and run its methods (see train.py)
        framework = CaloFastSim(data_path=data_path, device=device, reload_model=reload_model, **parameters, parameters=parameters)
        framework.training()
        framework.plot_history()
        framework.plot_learning_rates()
        framework.evaluate_test_loss()
        framework.sample_data()


    elif ( mode == 'evaluate'):

        # Instantiate evaluation class and run its methods (see evaluation.py)
        framework_evaluation = Judge(run_id = parameters["run_id"], sample_size = args.samplesize, data_path = data_path, device = device, parameters=parameters, thickness_bin = args.thickness_bin, distance_bin=args.distance_bin, bin_comparison_thickness = args.compare_bins_thickness, bin_comparison_distance=args.compare_bins_distance, generate=args.generate_samples)
        framework_evaluation.plot_averages()
        framework_evaluation.plot_histograms()
        if args.count_clusters: framework_evaluation.cluster_shower_images()
        framework_evaluation.show_correlation_matrix()
        framework_evaluation.comparison_images()

    elif ( mode == 'classifier'):

        # Instantiate evaluation class and run the classifier (see evaluation.py & classifier.py)
        framework_evaluation = Judge(run_id = parameters["run_id"], sample_size = args.samplesize, data_path = data_path, device = device, parameters=parameters, thickness_bin = args.thickness_bin, distance_bin=args.distance_bin, bin_comparison_thickness = args.compare_bins_thickness, bin_comparison_distance=args.compare_bins_distance, generate=args.generate_samples)
        framework_evaluation.train_binary_classifier(classifier_type=args.type_classifier)