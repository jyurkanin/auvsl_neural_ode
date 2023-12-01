cmake_minimum_required (VERSION 2.8.11)
project(auvsl_dynamics_bptt)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PythonLibs 3.8)


#set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
#set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -Ofast -mtune=native -march=native -DNDEBUG")

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")

#add_compile_options(
#  "-Wall" "-Ofast" "-mtune=native" "-march=native" "-DNDEBUG"
#)

add_compile_options(
  "-Wall"
  "$<$<CONFIG:Debug>:-O0;-g3;-ggdb3>"
  "$<$<CONFIG:Release>:-Ofast;-mtune=native;-march=native;-DNDEBUG>"
)

#add_compile_options(-fsanitize=undefined)
#add_link_options(-fsanitize=undefined -static-libasan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(/usr/include/eigen3)

include_directories(include/auvsl_dynamics)
include_directories(src)
include_directories(lib)

include(GNUInstallDirs)



include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  )
FetchContent_MakeAvailable(googletest)
enable_testing()



add_executable(train
	src/train.cpp
	src/VehicleSystem.cpp
	src/Trainer.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/BekkerDynamics.cpp
	src/auvsl_dynamics/BekkerTireModel.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(train PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(train ${PYTHON_LIBRARIES})


add_executable(train_bekker
	src/train_bekker.cpp
	src/BekkerSystem.cpp
	src/Trainer.cpp
	src/BaseNetwork.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/BekkerDynamics.cpp
	src/auvsl_dynamics/BekkerTireModel.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(train_bekker PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(train_bekker ${PYTHON_LIBRARIES})


add_executable(evaluate_bekker
	src/evaluate_bekker.cpp
	src/BekkerSystem.cpp
	src/Trainer.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/BekkerDynamics.cpp
	src/auvsl_dynamics/BekkerTireModel.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(evaluate_bekker PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(evaluate_bekker ${PYTHON_LIBRARIES})



add_executable(evaluate
	src/evaluate.cpp
	src/VehicleSystem.cpp
	src/Trainer.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
 	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(evaluate PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(evaluate ${PYTHON_LIBRARIES})


add_executable(train_linear
	src/linear/train_linear.cpp
	src/linear/LinearTrainer.cpp
	src/linear/LinearSystem.cpp
)
target_include_directories(train_linear PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(train_linear ${PYTHON_LIBRARIES})


add_executable(gen_tire_data
	src/gen_tire_model_data.cpp
)



add_executable(
	plot_stuff
	src/plot_stuff.cpp
	src/VehicleSystem.cpp
	src/BekkerSystem.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
	src/Trainer.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/BekkerDynamics.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/BekkerTireModel.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(plot_stuff PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(plot_stuff ${PYTHON_LIBRARIES})
target_link_libraries(plot_stuff GTest::gtest_main)




add_executable(
	test_Vehicle
	test/test_Vehicle.cpp
	src/VehicleSystem.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(test_Vehicle PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_Vehicle ${PYTHON_LIBRARIES})
target_link_libraries(test_Vehicle GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_Vehicle)




add_executable(
	test_Terrain
	test/test_Terrain.cpp
	src/VehicleSystem.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(test_Terrain PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_Terrain ${PYTHON_LIBRARIES})
target_link_libraries(test_Terrain GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_Terrain)




add_executable(
	test_TireNetwork
	test/test_TireNetwork.cpp
	src/VehicleSystem.cpp
	src/TestTerrainMaps.cpp
	src/BaseNetwork.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(test_TireNetwork PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_TireNetwork ${PYTHON_LIBRARIES})
target_link_libraries(test_TireNetwork GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_TireNetwork)

add_executable(
	test_Bekker
	test/test_Bekker.cpp
	src/auvsl_dynamics/BekkerTireModel.cpp
	src/auvsl_dynamics/utils.cpp
	src/BaseNetwork.cpp
	src/BekkerSystem.cpp
	src/TestTerrainMaps.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/BekkerDynamics.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(test_Bekker PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_Bekker ${PYTHON_LIBRARIES})
target_link_libraries(test_Bekker GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_Bekker)


add_executable(
	test_Trainer
	test/test_Trainer.cpp
	src/Trainer.cpp
	src/VehicleSystem.cpp
	src/BekkerSystem.cpp
	src/BaseNetwork.cpp
	src/TestTerrainMaps.cpp
	src/auvsl_dynamics/BekkerDynamics.cpp
	src/auvsl_dynamics/HybridDynamics.cpp
	src/auvsl_dynamics/TireNetwork.cpp
	src/auvsl_dynamics/BekkerTireModel.cpp
	src/auvsl_dynamics/utils.cpp
	src/auvsl_dynamics/generated/forward_dynamics.cpp
	src/auvsl_dynamics/generated/inertia_properties.cpp
	src/auvsl_dynamics/generated/inverse_dynamics.cpp
	src/auvsl_dynamics/generated/jacobians.cpp
	src/auvsl_dynamics/generated/jsim.cpp
	src/auvsl_dynamics/generated/miscellaneous.cpp
	src/auvsl_dynamics/generated/transforms.cpp
)
target_include_directories(test_Trainer PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_Trainer ${PYTHON_LIBRARIES})
target_link_libraries(test_Trainer GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_Trainer)



