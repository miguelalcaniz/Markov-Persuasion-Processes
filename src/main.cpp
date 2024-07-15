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
  std::string DATA = source + filename;

  // Reading the Markov Persuasion Process variables

  read_enviroment(env, DATA);

  // Printing the Markov Persuasion Process variables

  print_enviroment(env);

  // Algorithm 1 (Sender-Receivers Interaction)
  std::cout<< "------------------------------------------------\n";
  std::cout<< "TEST OF ALGORITHM 1 (Sender-Receivers Interaction)\n";
  std::cout<< "------------------------------------------------\n";
  sign_scheme phi;
  phi.init_scheme(states, A);

  episode ep = S_R_interaction(env, phi);

  std::cout<< ep << std::endl << std::endl;

  // Algorithm 2 (Optimistic Persuasive Policy Search (full))

  std::cout<< "------------------------------------------------\n";
  std::cout<< "TEST OF ALGORITHM 2, Optimistic Persuasive Policy Search\n";
  std::cout<< "------------------------------------------------\n";
    
  unsigned int T = 1000; // T correspond to the number of iterations of the Optimistic Persuasive Policy Search algorithm
  
  Estimators est = OPPS(env, T);
  
  print_estimators(est);

  return 0;
}