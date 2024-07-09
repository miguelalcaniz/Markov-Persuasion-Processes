#include "markov_persuasion_process.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>



// This function reads a .txt file and sets all the information of the enviroment into the variables 


void read_enviroment(size_t &L, std::vector<int> &states, size_t &a, transitions &trans, 
                     rewards &Srewards, rewards &Rrewards, size_t &w, prior &mu, const std::string& fileName){
    
   // Initializing the .txt stream reader

   std::ifstream inputFile(fileName);

   if(!inputFile) {
      std::cerr << "The file couldn't be opened." << std::endl;
      return;
   }
  

   // Reading the number of states

   std::string textLine;
   std::getline(inputFile, textLine);

   inputFile >> L;
   states.resize(L);

   for(int i = 0; i < L; ++i)
       inputFile >> states[i];


   // Reading the number of actions   

   for(int i = 0; i < 3; ++i)
      std::getline(inputFile, textLine);

   inputFile >> a;

   for(int i = 0; i < 3; ++i)
      std::getline(inputFile, textLine);
   
   trans.init_transitions(states, a);
       

   // Reading the probability transitions

   int nStatesO, nStatesD;
   for(int l = 0; l < L-1; l++){
      nStatesO = states[l];
      nStatesD = states[l+1];
      for(int sO = 0; sO < nStatesO; sO++){
        for(int action = 0; action < a; action++){
          double p;
          std::vector<double> probs;
          for(int sD = 0; sD < nStatesD; sD++){
            inputFile >> p;
            probs.push_back(p);
          }
        trans.set_transitions(l, sO, action, probs); 
        }
      }
   }
  

  // Reading Omega

  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);
    
  inputFile >> w;
  

  // Reading the rewards of the sender 
  
  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);
  
  Srewards.init_rewards(states, w, a);
 
  double r;
  for(int l = 0; l < L-1; ++l){
    for(int s = 0; s < states[l]; ++s){
      for(int o = 0; o < w; ++o){
         std::vector<double> v;
         for(int i = 0; i < a; ++i){
           inputFile >> r;
           v.push_back(r);
         }
         Srewards.set_rewards(l, s, o, v);
      }
    }
  }
  

  // Reading the rewards of the receiver 
  
  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);
  
  Rrewards.init_rewards(states, w, a);
 
  for(int l = 0; l < L-1; ++l){
    for(int s = 0; s < states[l]; ++s){
      for(int o = 0; o < w; ++o){
         std::vector<double> v;
         for(int i = 0; i < a; ++i){
           inputFile >> r;
           v.push_back(r);
         }
         Rrewards.set_rewards(l, s, o, v);
      }
    }
  }
  

  // Reading the prior function, the probability of the outcomes given a state

  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);

  mu.init_prior(states, w);
    
  double p;
  for(int l = 0; l < L; l++){
    int sMax = states[l];
    for(int s = 0; s < sMax; ++s){
      std::vector<double> ps;
      for(int o = 0; o < w; ++o){
        inputFile >> p;
        ps.push_back(p);
      }
      mu.set_prior(l,s,ps);
    }
  }
  
  inputFile.close();

};




void print_enviroment(std::vector<int> &states, size_t &a, transitions &trans, 
                     rewards &Srewards, rewards &Rrewards, size_t &w, prior &mu){


  std::cout<< "--------------------------------------" << std::endl;
  std::cout<< "TESTING THE CLASS EPISODE" << std::endl << std::endl;

  // Printing the states

  size_t L = states.size();
  std::cout<< "The lenght of the episodes L is : " << L << "\n";
  std::cout<< "And every partition has the following numbers of states: ";
  for(int i = 0; i < L; ++i)
      std::cout<< states[i] << ' ';
  std::cout<< std::endl<< std::endl;

  // Printing the number of actions

  std::cout<< "The number of actions is : " << a << std::endl << std::endl;
   
  // Printing the probability transitions

  std::cout<< trans << std::endl;

  // Printing the number of outcomes

  std::cout<< "The number of outcomes is " << w << std::endl << std::endl;

  // Printing the sender rewards

  std::cout<< Srewards << std::endl;

  // Printing the receiver rewards

  std::cout<< Rrewards << std::endl;

  // Printing the prior function

  std::cout<< mu << std::endl;

};


std::ostream &operator<<(std::ostream &stream, episode &ep)
{
  stream << "The episode is:\n\n";
  for(int l = 0; l < ep.L-1; ++l){
    stream << '(' << ep.get_soa(l).getX() << ", " << ep.get_soa(l).getW() << ", ";
    stream << ep.get_soa(l).getA() << ')';
    if(l != ep.L-2) std::cout<< " --> ";
  }
  stream << std::endl;
  return stream;
};


