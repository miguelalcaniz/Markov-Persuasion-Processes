#ifndef OPTOPT_HPP
#define OPTOPT_HPP

#include <vector>
#include <utility>
#include <random>

// Alias por std::vector<double>
using probs = std::vector<double>;
using outcome_probs = std::vector<probs>; 
using state_outcome_probs = std::vector<std::vector<outcome_probs>>;


class sign_scheme{
public:

  //Constructor
  sign_scheme(){};

  //Method for initializing the vector with size L
  void init_scheme(const std::vector<int>& states_values,const size_t a_value, const size_t w_value){
    
    L = states_values.size();
    states = states_values;
    a = a_value;
    w = w_value;

    squeme.resize(L);

    for(int l = 0; l < L; ++l){
      squeme[l].resize(states[l]);
      for(int s = 0; s < states[l]; ++s){
          squeme[l][s].resize(w);
          for(int k = 0; k < w; ++k){
              squeme[l][s][k].resize(a);
          }
      }
    }
  }

  double get_sign(int l, int s, int k, int a_value){
    return squeme[l][s][k][a_value];
  }


  int recommendation(int l, int s, int outcome){
         
    std::random_device re;
    std::knuth_b knuth(re());
   
    probs &prob = squeme[l][s][outcome];
    std::discrete_distribution<> distribution(prob.begin(), prob.end());
    return distribution(knuth);
  }

  // Declaration of Optimistic optimitation problem 
  // (definition in .cpp file)
  friend void OptOpt(sign_scheme &SQ);


private:
  size_t L;
  size_t w;
  size_t a;
  std::vector<int> states;
  state_outcome_probs squeme ;
};


#endif /* OPTOPT_HPP */