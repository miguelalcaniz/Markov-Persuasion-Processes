#include "episode_generator.hpp"
#include <vector>
#include <iostream>
#include <sstream>
#include <utility>
#include <map>


/// Definition of the stream operator for the transitions class

std::ostream &operator<<(std::ostream &stream, transitions &trans)
{
   stream << "The probability transitions are:\n\n";
   int L = trans.L;
   int nStatesO;
   for(int l = 0; l < L-1; l++){
      nStatesO = trans.states[l];
      for(int sO = 0; sO < nStatesO; sO++){
        for(int action = 0; action < trans.a; action++){
         for(auto const& el: trans.get_transitions(l, sO, action))
           stream << el << ' ';
         stream << std::endl;
        }
      }
      stream << std::endl;
   }
  return stream;
}


/// Definition of the stream operator for the rewards class

std::ostream &operator<<(std::ostream &stream, rewards &rewards){

  stream << "The sender rewards are:\n\n";

  for(int l = 0; l < rewards.L-1; ++l){
   for(int s = 0; s < rewards.states[l]; ++s){
      for(int o = 0; o < rewards.w; ++o){
         for(int i = 0; i < rewards.a; ++i)
           stream << rewards.get_reward(l, s, o, i) << " ";
         stream << "\n";
      }
   }
   stream << std::endl;
  }
  return stream;
}


/// Definition of the stream operator for the prior class

std::ostream &operator<<(std::ostream &stream, prior &mu){

  stream << "The prior function is:\n\n";

  for(int l = 0; l < mu.L; l++){
    int sMax = mu.states[l];
    for(int s = 0; s < sMax; ++s){
      for(int o = 0; o < mu.w; ++o)
        stream << mu.get_prior(l,s,o) << ' ';
      stream << std::endl;
    }
    stream << std::endl;
  }

  return stream;

}
 
