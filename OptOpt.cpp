#include "OptOpt.hpp"
#include <vector>

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
          for(int k = 0; k < SQ.w; ++k){
              double sum = 0;
              double value;
              std::vector<double> v(SQ.a);

              for(int r = 0; r < SQ.a; ++r){
                value = dis(knuth);
                sum += value;
                SQ.squeme[l][s][k][r] = value;
              }
              // Normalize the squeme because it corresponds to a probability distribution
              for(int r = 0; r < SQ.a; ++r)
                SQ.squeme[l][s][k][r] /= sum;
          }
      }
    }
}