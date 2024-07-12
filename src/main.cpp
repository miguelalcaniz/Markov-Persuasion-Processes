#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include "episode_generator.hpp"
#include "markov_persuasion_process.hpp"
#include "OptOpt.hpp"


int main() {

  // Declaring the Markov Persuasion Process variables

  size_t L;
  std::vector<int> states;
  size_t A;
  transitions trans;
  rewards<TypeReward::Sender> Srewards;
  rewards<TypeReward::Receiver> Rrewards; 
  prior mu;

  std::string source = "././test/";
  std::string filename = "data.txt";
  //std::string filename = "data2.txt"; ???????????????????'
  std::string DATA = source + filename;

  // Reading the Markov Persuasion Process variables

  read_enviroment(L, states, A, trans, Srewards, Rrewards, mu, DATA);

  // Printing the Markov Persuasion Process variables

  print_enviroment(states, A, trans, Srewards, Rrewards, mu);

  // Initializing and printing signaling squeme

  sign_scheme phi;
  phi.init_scheme(states, A);

  OptOpt(phi);

    for(int l = 0; l < L; ++l){
      for(int s = 0; s < states[l]; ++s){
          for(int k = 0; k < A; ++k){
              for(int r = 0; r < A; ++r)
                std::cout<< phi.get_sign(l,s,k,r) << ' ';
              std::cout<< std::endl;
          }
          std::cout<< std::endl;
      }
      std::cout<< std::endl;
    }


  // Algorithm 1 (Sender-Receiver Interaction at episode t)

  int actual_state = 0;
  int outcome;
  int action;
  
  episode ep(L);

  std::cout<< "HERE WE TEST ALGORITHM 1 (Sender-Receiver Interaction) \n\n";

  for(int l = 0; l < L-1; ++l){
    outcome = mu.generate_outcome(l, actual_state);
    action = phi.recommendation(l, actual_state, outcome);
    SOA soa(l, outcome, action);
    ep.set_soa(l, soa);
    actual_state = trans.next_state(l, actual_state, action);
  }

 std::cout<< ep << std::endl;

 
  // Testing the prior function and its method to randomly generate an outcome
/*
  std::cout<< "Imprimiendo el vector de probabilidades de siguientes estados:\n\n";
  std::vector<double> v = mu.get_prior(0,0);
  for(const auto &el: v) std::cout<< el << ' ';
  std::cout<< std::endl << std::endl;

  int n = 20;
  for(int i = 0; i < n; ++i){
    std::cout<< mu.generate_outcome(0,0) << std::endl;
  }
  std::cout<< std::endl;
*/

  return 0;
}