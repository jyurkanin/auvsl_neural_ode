#include "VehicleSystem.h"
#include "Trainer.h"





int main()
{
  Trainer train;

  for(int i = 0; i < 1000; i++)
  {
    //train.train();
    train.evaluate();
  }
  
}
