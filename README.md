# Parametrizing Fast Calorimeter Simulations for Variable Detector Conditions

This repository contains the code of my Bachelor's thesis project with the title 'Parametrizing Fast Calorimeter Simulations for Variable Detector Conditions'. It aims at increasing the flexibility of calorimeter simulations by parametrizing the generative model on various detector conditions. <br>
As underlying model, we use autoregressive normalizing flows with rational quadratic spline transformations.<br>
<br>
To train the model, we run

`python3 main.py -m train -r [specific run id] --device [GPU_device]`

giving the training a run identification.<br>
<br>
To evaluate the model, we run

`python3 main.py -m evaluate -r [specific run id] -s [sample size] --device [GPU_device]...`

again specifying the run id to evaluate as well as the sample size of the analyzed samples.


> Note: More information might be added later or can be found in the final thesis.
