#include "OptOpt.hpp"
#include <vector>


//Method for initializing the vector with size L
void sign_scheme::init_scheme(const std::vector<int>& states_values,const size_t A_value)
{   
  L = states_values.size();
  states = states_values;
  A = A_value;

  squeme.resize(L);

  for(int l = 0; l < L; ++l){
    squeme[l].resize(states[l]);
    for(int s = 0; s < states[l]; ++s){
      squeme[l][s].resize(A);
      for(int k = 0; k < A; ++k)
        squeme[l][s][k].resize(A);
    }
  }
}

// Method for getting the recommendation out of the probability distribution
int sign_scheme::recommendation(int l, int s, int outcome)
{      
  std::random_device re;
  std::knuth_b knuth(re());
 
  probs &prob = squeme[l][s][outcome];
  std::discrete_distribution<> distribution(prob.begin(), prob.end());
  int rec = distribution(knuth);
  return rec;
}


/*
This function OptOpt is supposed to solve a maximitation problem called Optimistic Optimitation problem (Opt-Opt)
*/

void OptOpt(sign_scheme &SQ){

    size_t L = SQ.states.size();
    
    std::random_device re;
    std::knuth_b knuth(re());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int l = 0; l < L; ++l){
      for(int s = 0; s < SQ.states[l]; ++s){
          for(int k = 0; k < SQ.A; ++k){
              double sum = 0;
              double value;
              std::vector<double> v(SQ.A);

              for(int r = 0; r < SQ.A; ++r){
                value = dis(knuth);
                sum += value;
                SQ.squeme[l][s][k][r] = value;
              }
              // Normalize the squeme because it corresponds to a probability distribution
              for(int r = 0; r < SQ.A; ++r)
                SQ.squeme[l][s][k][r] /= sum;
          }
      }
    }
}