#include "LinearSystem.h"
#include "LinearTrainer.h"

#include <iostream>
#include <fenv.h>



int main()
{
	LinearTrainer train;
	std::cout << "Default Performance:\n";
	
	train.load();
	
	// train.evaluate_cv3();
	// train.evaluate_ld3();
	// train.evaluate_train3();
	
	for(int i = 0; i < 10000; i++)
	{
		train.train();
		train.save();    
		
		if((i % 40) == 39)
		{
			train.evaluate_cv3();
			train.evaluate_ld3();
			return 0;
		}
	}
    
}
