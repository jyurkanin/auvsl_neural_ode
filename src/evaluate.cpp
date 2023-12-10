#include "VehicleSystem.h"
#include "TestTerrainMaps.h"
#include "Trainer.h"
#include <iostream>

int main()
{
	int num_threads = 1;

	std::shared_ptr<const FlatTerrainMap<ADF>> map = std::make_shared<const FlatTerrainMap<ADF>>();
	std::shared_ptr<VehicleSystemFactory<ADF>> factory = std::make_shared<VehicleSystemFactory<ADF>>(map);
	Trainer train(factory, num_threads);
	train.load();
	
	train.evaluate_cv3();
	train.evaluate_ld3();
	train.evaluate_train3();	
}
