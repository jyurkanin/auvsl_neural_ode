#include "VehicleSystem.h"
#include "Trainer.h"
#include <iostream>
#include <fenv.h>



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
		train.trainThreads();
		train.evaluate_validation_dataset();
		// train.train();
		train.save();    
		
		// if((i % 40) == 39)
		// {
		// 	train.evaluate_cv3();
		// 	train.evaluate_ld3();
		// }
	}
    
}
