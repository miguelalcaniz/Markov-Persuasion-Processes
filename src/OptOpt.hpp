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
  std::vector<int> states;
  state_outcome_probs squeme ;
};


#endif /* OPTOPT_HPP */