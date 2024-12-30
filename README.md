# Parametrizing Fast Calorimeter Simulations for Variable Detector Conditions

This repository contains the information, code and results of my Bachelor's thesis project with the title 'Parametrizing Fast Calorimeter Simulations for Variable Detector Conditions'. It aims at increasing the flexibility of calorimeters simulations by parametrizing the generative model on various detector conditions. <br>
As underlying model, we use autoregressive normalizing flows with rational quadratic spline transformations.<br>
<br>
To train the model, we run

`python3 -m calo_sim.[enter_condition].main -m train -p [particle type] -r [specific run id]`

specifying the detector condition, the particle which we want to simulate ($\pi_0$, $\gamma$ or both) and giving the training a run identification.<br>
<br>
To evaluate the model, we run

`python3 -m calo_sim.[enter_condition].main -m evaluate -r [specific run id] -s [sample size]`

again specifying the run id to evaluate as well as the sample size of the analyzed samples.

## First condition: Detector Noise

As first detector condition, we imitate the underlying electronic noise in the CMS ECAL by adding Gaussian noise to the calorimeter crystals. The standard deviation stems from a $[10, 400]~\mathrm{MeV}$ range.<br>
We get the displayed results of the generation

![Samples of Generated Images](calo_sim/results/samples_50_photon.png)



> Note: More information might be added later or can be found in the final thesis.