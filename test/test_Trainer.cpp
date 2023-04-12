#include <vector>
#include "gtest/gtest.h"

#include <cpp_bptt.h>
#include <cpp_neural.h>
#include "Trainer.h"

namespace {
  TEST(TrainerSet, load_save)
  {
    std::string file_name = "temp.txt";
    VectorAD init_params(100);
    VectorAD temp_params(100);
    for(int i = 0; i < init_params.size(); i++)
    {
      init_params[i] = i;
    }
    
    Trainer trainer;
    trainer.saveVec(init_params, file_name);
    trainer.loadVec(temp_params, file_name);

    for(int i = 0; i < temp_params.size(); i++)
    {
      EXPECT_EQ(init_params[i], temp_params[i]);
    }
  }
}
