cmake_minimum_required (VERSION 2.8.11)
project(auvsl_dynamics_bptt)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PythonLibs 3.8)


set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -O3 -mtune=native -march=native -DNDEBUG")

add_compile_options(
  "-Wall" "-O3" "-mtune=native" "-march=native" "-DNDEBUG"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(/usr/include/eigen3)
include_directories(/home/justin/code/CppNeural/src/cpp_neural_lib/src)
include_directories(/home/justin/code/CppNeural/src/cpp_bptt_lib/src)

include_directories(include/auvsl_dynamics)
include_directories(src)
include_directories(/home/justin/code/matplotlib-cpp)

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
	
	/home/justin/code/CppNeural/src/cpp_bptt_lib/src/SimulatorAD.cpp
)

target_include_directories(train PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(train ${PYTHON_LIBRARIES})

add_executable(
	runUnitTest
	test/test_Vehicle.cpp
	src/VehicleSystem.cpp
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

	/home/justin/code/CppNeural/src/cpp_bptt_lib/src/SimulatorAD.cpp
)
target_include_directories(runUnitTest PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(runUnitTest ${PYTHON_LIBRARIES})
target_link_libraries(runUnitTest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(runUnitTest)


add_executable(
	test_TireNetwork
	test/test_TireNetwork.cpp
	src/VehicleSystem.cpp
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

	/home/justin/code/CppNeural/src/cpp_bptt_lib/src/SimulatorAD.cpp
)
target_include_directories(test_TireNetwork PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_TireNetwork ${PYTHON_LIBRARIES})
target_link_libraries(test_TireNetwork GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_TireNetwork)


add_executable(
	test_Trainer
	test/test_Trainer.cpp
	src/Trainer.cpp
	src/VehicleSystem.cpp
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

	/home/justin/code/CppNeural/src/cpp_bptt_lib/src/SimulatorAD.cpp
)
target_include_directories(test_Trainer PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_Trainer ${PYTHON_LIBRARIES})
target_link_libraries(test_Trainer GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_Trainer)
