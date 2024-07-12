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

  Enviroment env;

  // Defining some references to the variables to avoid writing env.
  size_t& L = env.L;
  TensorI& states = env.states;
  size_t& A = env.A;
  transitions& trans = env.trans;
  rewards<TypeReward::Sender>& Srewards = env.Srewards;
  rewards<TypeReward::Receiver>& Rrewards = env.Rrewards; 
  prior& mu = env.mu;

  // Setting the the data file name and the data file source

  std::string source = "././test/";
  std::string filename = "data.txt";
  //std::string filename = "data2.txt";    STILL IN THE TO DO LIST 
  std::string DATA = source + filename;

  // Reading the Markov Persuasion Process variables

  read_enviroment(env, DATA);

  // Printing the Markov Persuasion Process variables

  print_enviroment(env);

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


  // Algorithm 1 (Sender-Receivers Interaction at episode t)

 episode ep = S_R_interaction(env, phi);

 std::cout<< ep << std::endl;

  // Algorithm 2 (Optimistic Persuasive Policy Search (full))
  
  // Require: X, A, T, confidence parameter delta ( 0 < delta < 1)
  unsigned int T = 10;

  // Initialize all estimators to 0
  prior estimated_mu;
  transitions estimated_trans;
  rewards<TypeReward::Sender> estimated_SR;
  rewards<TypeReward::Receiver> estimated_RR;


  // Initialize all bounds to +inf

  return 0;
}