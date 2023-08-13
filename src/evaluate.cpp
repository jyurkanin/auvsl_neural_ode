#include "VehicleSystem.h"
#include "Trainer.h"
#include <iostream>

int main()
{
	int num_threads = 1;
	
	Trainer train(num_threads);
	train.load();
	
	train.evaluate_cv3();
	train.evaluate_ld3();
	train.evaluate_train3();	
}
