#ifndef OPTOPT_HPP
#define OPTOPT_HPP

#include <vector>
#include <utility>
#include <random>
#include <iostream>


// Alias for making the code more readable
using TensorI = std::vector<int>;
using TensorD = std::vector<double>;
using Tensor2D = std::vector<TensorD>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;

class sign_scheme{
public:

  // Constructor
  sign_scheme(){};

  // Method for initializing the vector with size L
  void init_scheme(const std::vector<int>& states_values,const size_t A_value);

  // Method for getting the values
  double get_sign(int l, int s, int k, int a_value){
    return squeme[l][s][k][a_value];
  }

  // Method for getting the recommendation out of the probability distribution
  int recommendation(int l, int s, int outcome);

  // Declaration of Optimistic optimitation problem 
  friend void OptOpt(sign_scheme &SQ);


private:
  size_t L;
  size_t A;
  TensorI states;
  Tensor4D squeme ;
};


#endif /* OPTOPT_HPP */