# Noise Estimation

Adaptive Measurement Noise Estimation for Nonlinear Kalman Filtering

All the required external dependencies are listed in the requirements.txt file, thus allowing an easy installation with PIP as follows: "pip install -r requirements.txt"

The main estimation and simulation code resides in the "noiseestimation" package folder, most of it is also documented with the resulting Documentation in the "doc" folder.

The "linear", "ekf" and "ukf" folders contain all the scenarios for which the estimation methods were evaluated. In order to start them, they need to be called as python modules, eg. "python -m ukf.bicycle.filtering"

In the "data" folder there are all of the used vehicle readings, which were transformed into the present json file format using the scripts in the "utils" folder

The "sympy" folder holds several python files used for calculating the Jacobians needed for the Extended Kalman Filter scenarios.

Lastly the project's unit tests are found in the "tests" folder
