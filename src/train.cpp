#include "VehicleSystem.h"
#include "Trainer.h"
#include <iostream>

#include <fenv.h>


int main()
{
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
  
  Trainer train;
  std::cout << "Default Performance:\n";
  
  train.load();
  train.evaluate_cv3();
  train.evaluate_ld3();
  

  for(int i = 0; i < 10000; i++)
  {
    train.train();
    train.save();
    

    if((i % 3) == 0)
    {
      train.evaluate_cv3();
      train.evaluate_ld3();
    }
  }
    
}
