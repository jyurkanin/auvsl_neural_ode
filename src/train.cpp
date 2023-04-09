#include "VehicleSystem.h"
#include "Trainer.h"





int main()
{
  Trainer train;
  train.evaluate_cv3();
  train.evaluate_ld3();
  for(int i = 0; i < 10; i++)
  {
    train.train();
    
  }
  train.evaluate_cv3();
  train.evaluate_ld3();
  
}
