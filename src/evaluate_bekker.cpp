#include "BekkerSystem.h"
#include "Trainer.h"
#include <iostream>

int main()
{
	int num_threads = 1;
	
	std::shared_ptr<BekkerSystemFactory<ADF>> factory = std::make_shared<BekkerSystemFactory<ADF>>();
	Trainer train(factory, num_threads);
	train.load();
	
	train.evaluate_cv3();
	train.evaluate_ld3();
	train.evaluate_train3();	
}
