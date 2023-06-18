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
  // train.evaluate_train3();

  for(int i = 0; i < 10000; i++)
  {
    train.train();
    train.save();    
	
    if((i % 20) == 19)
    {
      train.evaluate_cv3();
      train.evaluate_ld3();
    }
  }
    
}
