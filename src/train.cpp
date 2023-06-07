#include "VehicleSystem.h"
#include "Trainer.h"
#include <iostream>




int main()
{
  int num_threads = 4;
    
  Trainer train(num_threads);
  std::cout << "Default Performance:\n";
  
  train.load();
  // train.evaluate_cv3();
  // train.evaluate_ld3();
  

  for(int i = 0; i < 1; i++)
  {
    train.trainThreads();
    train.save();
    

    if((i % 5) == 4)
    {
      train.evaluate_cv3();
      train.evaluate_ld3();
    }
  }
    
}
