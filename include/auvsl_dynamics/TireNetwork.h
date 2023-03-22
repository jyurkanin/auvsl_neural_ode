#pragma once
#include <fstream>
#include <string.h>
#include <Eigen/Dense>

#include "generated/forward_dynamics.h"

using Jackal::rcg::Scalar;

//simple feedforward network with hard coded num of layers
//THis could have been a single function honestly
class TireNetwork {
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  TireNetwork();
  ~TireNetwork();

  static const int num_hidden_nodes = 16;
  static const int num_in_features = 5;
  static const int num_out_features = 3;

  static int  getNumParams();
  static void setParams(const VectorS &params, int idx);
  static void getParams(VectorS &params, int idx);
  static void forward(const Eigen::Matrix<Scalar,num_in_features,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec);
  
  static int m_is_loaded;
  static void load_model();
  
  static Eigen::Matrix<Scalar,num_hidden_nodes,num_in_features> m_weight0;
  static Eigen::Matrix<Scalar,num_hidden_nodes,1> m_bias0;
  static Eigen::Matrix<Scalar,num_hidden_nodes,num_hidden_nodes> m_weight2;
  static Eigen::Matrix<Scalar,num_hidden_nodes,1> m_bias2;
  static Eigen::Matrix<Scalar,num_out_features,num_hidden_nodes> m_weight4;
  static Eigen::Matrix<Scalar,num_out_features,1> m_bias4;

private:
  static Eigen::Matrix<Scalar,num_out_features,1> m_out_mean;
  static Eigen::Matrix<Scalar,num_out_features,1> m_out_std;
  static Eigen::Matrix<Scalar,num_in_features,1>  m_in_mean;
  static Eigen::Matrix<Scalar,num_in_features,1>  m_in_std_inv; //inverse of in_std. Because multiply is faster than divide.

};
