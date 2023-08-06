#pragma once
#include <fstream>
#include <string.h>
#include <Eigen/Dense>

#include "generated/forward_dynamics.h"

using Jackal::rcg::Scalar;



//simple feedforward network with hard coded num of layers
//THis could have been a single function honestly
class TireNetwork
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  TireNetwork();
  ~TireNetwork();

  static const int num_hidden_nodes = 8;
  static const int num_hidden_nodes2 = 8;
  static const int num_in_features = 4;
  static const int num_out_features = 4;
  static const int num_networks = 4;
  
  int  getNumParams();
  void setParams(const VectorS &params, int idx);
  void getParams(VectorS &params, int idx);
  void forward(const Eigen::Matrix<Scalar,8,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec, int ii);
  
  int is_loaded;
  int load_model();
  

  struct Params
  {
	  Eigen::Matrix<Scalar,num_hidden_nodes,3> weight0;
	  Eigen::Matrix<Scalar,num_hidden_nodes,num_hidden_nodes> weight2;
	  Eigen::Matrix<Scalar,2,num_hidden_nodes> weight4;
	  
	  Eigen::Matrix<Scalar,num_hidden_nodes2,1> z_weight0;
	  Eigen::Matrix<Scalar,num_hidden_nodes2,1> z_bias0;
	  Eigen::Matrix<Scalar,num_hidden_nodes2,num_hidden_nodes2> z_weight2;
	  Eigen::Matrix<Scalar,num_hidden_nodes2,1> z_bias2;
	  Eigen::Matrix<Scalar,1,num_hidden_nodes2> z_weight4;
	  Eigen::Matrix<Scalar,1,1> z_bias4;
  };

  Params m_params[4];
  
private:
  Eigen::Matrix<Scalar,3,1> out_std;
  Eigen::Matrix<Scalar,num_in_features,1>  in_mean;
  Eigen::Matrix<Scalar,num_in_features,1>  in_std_inv; //inverse of in_std. Because multiply is faster than divide.

};
