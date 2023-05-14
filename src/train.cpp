#include "VehicleSystem.h"
#include "Trainer.h"
#include <iostream>




int main()
{
  Trainer train;
  std::cout << "Default Performance:\n";
  
  train.load();
  //train.evaluate_cv3();
  train.evaluate_ld3();
  
  for(int i = 0; i < 1000; i++)
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
