#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>

#include "bekker_model.cpp" //lazy

int num_in_features = 8;

float rand_float(float max, float min){
  return ((max - min)*((float)rand()/RAND_MAX)) + min;
}



std::vector<float> get_forces(float vx, float vy, float w,
			      float zr, float bk_kc, float bk_kphi,
			      float bk_n0, float bk_n1, float bk_phi)
{
  float vel_x_tan = 0.098 * w;
  float slip_ratio;  // longitudinal slip
  float slip_angle;  // lateral

  std::vector<float> features(8);
  features[3] = bk_kc;
  features[4] = bk_kphi;
  features[5] = bk_n0;
  features[6] = bk_n1;
  features[7] = bk_phi;  

  if(vel_x_tan != 0.0f)
  {
    slip_ratio = fabs(vel_x_tan - vx)/fabs(vel_x_tan);
  }
  else
  {
    slip_ratio = fabs(vel_x_tan - vx)/1e-3;
  }
  if(vx != 0.0)
  {
    slip_angle = atan(fabs(vy) / fabs(vx));  
  }
  else
  {
    slip_angle = atan(fabs(vy) / 1e-3);  
  }
  
  
  features[0] = zr;
  features[1] = slip_ratio; //1 - (vx/(.1*wy));
  features[2] = slip_angle;
  
  std::vector<float> forces(6);
  forces = tire_model_bekker(features);
  
  if(vel_x_tan > vx){
    forces[0] = fabs(forces[0]);
    forces[3] = fabs(forces[3]);
  }
  else{
    forces[0] = -fabs(forces[0]);
    forces[3] = -fabs(forces[3]);  
  }
  
  if(vy > 0){
    forces[1] = -fabs(forces[1]);
    forces[4] = -fabs(forces[4]);
  }
  else{
    forces[1] = fabs(forces[1]);
    forces[4] = fabs(forces[4]);
  }
  
  return forces;
}



int main(){
  std::ofstream nn_file("tire_data.csv");
  nn_file << "vx,vy,w,zr,kc,kphi,n0,n1,phi,Fx,Fy,Fz\n";
    
  std::vector<float> wrench;
  for(int i = 0; i < 1000000; i++)
  {
    float vx = rand_float(1, -1); // The bekker model is literally not fucking equipped to handle negative velocities
    float vy = rand_float(1, -1);
    float wy = rand_float(1, -1);
    float zr = 0.003; //rand_float(.01,.0001);

    //29.76, 2083, 0.8, 0, 0.392699
    float bk_kc   = 29.76; //rand_float(100.0,20.0);
    float bk_kphi = 2083.0; //rand_float(3500.0,500.0);
    float bk_n0   = 0.8; //rand_float(1.3,0.3);
    float bk_n1   = 0.0; //rand_float(0.2,0.0);
    float bk_phi  = 0.3927; //rand_float(0.52,0.17);
    
    wrench = get_forces(vx, vy, wy, zr, bk_kc, bk_kphi, bk_n0, bk_n1, bk_phi);
    
    nn_file << vx << ',' << vy << ',' << wy << ',' << zr << ','
	    << bk_kc << ',' << bk_kphi << ',' << bk_n0 << ',' << bk_n1 << ',' << bk_phi << ','
	    << wrench[3] << ',' << wrench[4] << ',' << wrench[5] << '\n';
    
    if((i % 1000) == 0)
    {
      std::cout << i << "\n";
    }
  }
  
  nn_file.close();
  
  return 0;
}








/*
int main_ignore(){
  test_rand();
  
  SimpleTerrainMap simple_map;
  JackalDynamicSolver::init_model(2);
  JackalDynamicSolver::set_terrain_map(&simple_map);
  
  JackalDynamicSolver solver;
  
  Eigen::Matrix<float,JackalDynamicSolver::num_in_features,1> features;
  
  BekkerData soil_params = lookup_soil_table(0);
  
  features[0] = .003;   //sinkage
  features[1] = 0;      //slip_ratio
  features[2] = .5;      //slip_angle
  
  features[3] = soil_params.kc;
  features[4] = soil_params.kphi;         //kphi
  features[5] = soil_params.n0;           //n0
  features[6] = soil_params.n1;             //n1
  features[7] = soil_params.phi;

  
  SpatialVector bk;
  SpatialVector nn;
  
  std::ofstream force_log;
  force_log.open("/home/justin/code/AUVSL_ROS/src/auvsl_planner/data/f_tire_train.csv", std::ofstream::out);
  force_log << "zr,slip_ratio,slip_angle,kc,kphi,n0,n1,phi,Fx,Fy,Fz,Ty\n";
  
  for(unsigned i = 0; i < 100000; i++){
    features[0] = rand_float(.1,.00001);        //zr
    features[1] = rand_float(1,-1);             //slip_ratio
    features[2] = rand_float(M_PI_2,-M_PI_2);   //slip_angle
    features[3] = rand_float(100,20);           //kc
    features[4] = rand_float(3500,500);         //kphi
    features[5] = rand_float(1.3,.3);           //n0
    features[6] = rand_float(.2,0);             //n1
    features[7] = rand_float(.524,.175);

    
    bk = solver.tire_model_bekker(features);
    

    for(int j = 0; j < 8; j++){

      force_log << features[j] << ',';
    }
    
    force_log << bk[3] << ',';
    force_log << bk[4] << ',';
    force_log << bk[5] << ',';
    force_log << bk[1] << '\n';
  }

  
  force_log.close();

  JackalDynamicSolver::del_model();

  return 0;
}
*/
