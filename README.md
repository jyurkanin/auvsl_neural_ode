# Code for the paper: Differentiable Simulation for Data Driven Modelling of an Off-Road Skid-Steer Vehicle 
Using real Jackal vehicle trajectories, we can train a differentiable vehicle simulation end-to-end.
The vehicle model contains a 3D rigid body dynamic solver combined with a tire-soil model.
A Bekker based vehicle model is used as the benchmark. A neural network is pretrained to approximate
the Bekker tire-soil model. Then the rigid body solver + neural network tire-soil model is trained
end-to-end. This is possible because I used CppAD<double> for all the math, which lets you
automatically take derivatives. Also, I used robcogen to automatically generate the vehicle dynamic model.
You will have to install robcogen and configure it to use CppAD if you want to run this repo. Sorry.
