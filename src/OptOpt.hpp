#ifndef OPTOPT_HPP
#define OPTOPT_HPP

#include <vector>
#include <utility>
#include <random>
#include <iostream>

// Alias por std::vector<double>
using probs = std::vector<double>;
using outcome_probs = std::vector<probs>; 
using state_outcome_probs = std::vector<std::vector<outcome_probs>>;


class sign_scheme{
public:

  //Constructor
  sign_scheme(){};

  //Method for initializing the vector with size L
  void init_scheme(const std::vector<int>& states_values,const size_t A_value){
    
    L = states_values.size();
    states = states_values;
    A = A_value;

    squeme.resize(L);

    for(int l = 0; l < L; ++l){
      squeme[l].resize(states[l]);
      for(int s = 0; s < states[l]; ++s){
          squeme[l][s].resize(A);
          for(int k = 0; k < A; ++k){
              squeme[l][s][k].resize(A);
          }
      }
    }
  }

  double get_sign(int l, int s, int k, int a_value){
    return squeme[l][s][k][a_value];
  }


  int recommendation(int l, int s, int outcome)
  {      
    std::random_device re;
    std::knuth_b knuth(re());
   
    probs &prob = squeme[l][s][outcome];
    std::discrete_distribution<> distribution(prob.begin(), prob.end());
    int rec = distribution(knuth);
    return rec;
  }

  // Declaration of Optimistic optimitation problem 
  friend void OptOpt(sign_scheme &SQ);


private:
  size_t L;
  size_t A;
  std::vector<int> states;
  state_outcome_probs squeme ;
};


#endif /* OPTOPT_HPP */