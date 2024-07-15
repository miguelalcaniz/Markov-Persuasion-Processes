#ifndef OPTOPT_HPP
#define OPTOPT_HPP

#include <vector>
#include <utility>
#include <random>
#include <iostream>


/* Alias vectors for making the code more readable */
using TensorI = std::vector<int>;
using TensorD = std::vector<double>;
using Tensor2D = std::vector<TensorD>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;


/*
  Signaling scheme class. 
  It gives the probability distribution over the finite set S of 
  signals for the receivers for every state and outcome. In our setting 
  we will take Ω = A. 
  So the reccomendation would be to choose action (a) in the tuple 
  state-outcome (l, s, o) (state s of partition l of S and outcome o)
  with probability squeme[l][s][o][a]. 
*/

class sign_scheme{
public:
  /* Default constructor used with init_scheme to initizalize the variables */
  sign_scheme(){};

  /* Method for initializing the vector with size L */
  void init_scheme(const std::vector<int>& states_values,const size_t A_value);

  /* Method for getting the value of the recommendation of action given tuple state-outcome */
  double get_sign(int l, int s, int o, int a_value){
    return squeme[l][s][o][a_value];
  }

  /* Method for getting the recommendation probability vector over the actions */
  int recommendation(int l, int s, int outcome);

  /* 
    Optimistic Optimitation Problem 
    Function not programmed. (Just a patch function is coded)
    Theorically it returns the optimal signaling scheme for maximizing sender's rewards.
    It consists in solving a maximitation linear problem which couldn't be done. 
    In this code it just returns a valid random signaling scheme.
  */ 
  friend void OptOpt(sign_scheme &SQ);

private:
    size_t L; // Number of partitions of states
    size_t A; // Number of possible actions
    TensorI states; // Vector of states of every partition of S
    Tensor4D squeme; // Tensor that keeps the signal probabilities for every state-outcome-action
};

#endif /* OPTOPT_HPP */