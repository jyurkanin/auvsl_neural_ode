#include <vector>
#include "gtest/gtest.h"

#include "VehicleSystem.h"
#include "Trainer.h"
#include "TestTerrainMaps.h"


namespace {
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorF;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;

	TEST(TrainerSet, load_save)
	{
		std::string file_name = "temp.txt";
		VectorAD init_params(100);
		VectorAD temp_params(100);
		for(int i = 0; i < init_params.size(); i++)
		{
			init_params[i] = i;
		}
		
		std::shared_ptr<const FlatTerrainMap<ADF>> map;
		Trainer trainer(std::make_shared<VehicleSystemFactory<ADF>>(map));
		trainer.saveVec(init_params, file_name);
		trainer.loadVec(temp_params, file_name);

		for(int i = 0; i < temp_params.size(); i++)
		{
			EXPECT_EQ(init_params[i], temp_params[i]);
		}
	}
}
