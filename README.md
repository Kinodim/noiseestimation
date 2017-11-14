# Noise Estimation

Adaptive Measurement Noise Estimation for Nonlinear Kalman Filtering

All the required external dependencies are listed in the requirements.txt file, thus allowing an easy installation with PIP as follows: "pip install -r requirements.txt"

The main estimation and simulation code resides in the "noiseestimation" package folder, most of it is also documented with the resulting Documentation in the "doc" folder.

The "linear", "ekf" and "ukf" folders contain all the scenarios for which the estimation methods were evaluated. In order to start them, they need to be called as python modules, eg. "python -m ukf.bicycle.filtering".
Often there is a file executing only the kalman filtering, one for each tested estimation method and one for the comparison of those methods using multiple runs, the ukf bicycle folder also contains scenarios for a continuous estimation, a filter variant incorporating a sensor bias into the state, simulations of varying noise covariances, continuous estimation that does not use the estimated noise matrices and the estimation of a second (adenauer) dataset.
For the UKF the ML estimation method may also be called the "direct approach" while the MAP estimation is also referred to as the "continuous estimation".
The EKF and the bicycle scenarios in general also provide their own filter class.

In the "data" folder there are all of the used vehicle readings, which were transformed into the present json file format using the scripts in the "utils" folder

The "sympy" folder holds several python files used for calculating the Jacobians needed for the Extended Kalman Filter scenarios.

Lastly the project's unit tests are found in the "tests" folder
