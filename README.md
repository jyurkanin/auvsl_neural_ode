This trains a hybrid physics based neural ode representing an off-road vehicle
Using a single thread on a cpu, we do this as efficiently as possible by:
      - Auto generating all derivatives instead of using operator overloading to auto-diff
      - Only generating relatively small derivatives of the system and using the chain rule to back prop through time
      - Using Robcogen to auto generate the code for the articulated body algorithm of the vehicle

* copy the vehicle code into the system class
* auto generate the simulator class
