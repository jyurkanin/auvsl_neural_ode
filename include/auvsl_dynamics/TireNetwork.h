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

  static const int num_hidden_nodes = 16;
  static const int num_in_features = 5;
  static const int num_out_features = 3;
  static const int num_networks = 4;
  
  static int  getNumParams();
  static void setParams(const VectorS &params, int idx);
  static void getParams(VectorS &params, int idx);
  static void forward(const Eigen::Matrix<Scalar,9,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec, int ii);
  
  static int is_loaded;
  static int load_model();
  

  struct Params
  {
    Eigen::Matrix<Scalar,num_hidden_nodes,num_in_features> weight0;
    Eigen::Matrix<Scalar,num_hidden_nodes,1> bias0;
    Eigen::Matrix<Scalar,num_hidden_nodes,num_hidden_nodes> weight2;
    Eigen::Matrix<Scalar,num_hidden_nodes,1> bias2;
    Eigen::Matrix<Scalar,num_out_features,num_hidden_nodes> weight4;
    Eigen::Matrix<Scalar,num_out_features,1> bias4;
  };

  static Params m_params[4];
  
private:
  static Eigen::Matrix<Scalar,num_out_features,1> out_std;
  static Eigen::Matrix<Scalar,num_in_features,1>  in_mean;
  static Eigen::Matrix<Scalar,num_in_features,1>  in_std_inv; //inverse of in_std. Because multiply is faster than divide.

};
