#include "markov_persuasion_process.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>



// Method that generates an episode with a given prior distribution, siganling scheme and probabilities of transitions
void episode::generate_episode(prior mu, sign_scheme phi, transitions trans)
{
    int actual_state = 0;
    int outcome;
    int action;
     
    for(int l = 0; l < L-1; ++l){
      outcome = mu.generate_outcome(l, actual_state);
      action = phi.recommendation(l, actual_state, outcome);
      SOA soa(l, outcome, action);
      ep[l] = soa;
      actual_state = trans.next_state(l, actual_state, action);
    }
}


// This function reads a .txt file and sets all the information of the enviroment into the variables 

void read_enviroment(size_t &L, TensorI &states, size_t &A, transitions &trans, 
                     rewards<TypeReward::Sender> &Srewards, rewards<TypeReward::Receiver> &Rrewards, 
                     prior &mu, const std::string& fileName)
{
   // Initializing the .txt stream reader
   std::ifstream inputFile(fileName);

   if(!inputFile) {
      std::cerr << "The file couldn't be opened.\n";
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

   inputFile >> A;

   for(int i = 0; i < 3; ++i)
      std::getline(inputFile, textLine);
   
   trans.init_transitions(states, A);
       

   // Reading the probability transitions

   int nStatesO, nStatesD;
   for(int l = 0; l < L-1; l++){
      nStatesO = states[l];
      nStatesD = states[l+1];
      for(int sO = 0; sO < nStatesO; sO++){
        for(int action = 0; action < A; action++){
          double p, sum = 0;
          TensorD probs;
          for(int sD = 0; sD < nStatesD; sD++){
            inputFile >> p;
            if(p > 1 || p < 0){
              std::cerr << "The inputs given for partition " << l << " and state " << sO;
              std::cerr << " doesn't correspond to a probability distribution.\n";
              return; 
            } 
            sum += p;
            probs.push_back(p);
          }
          if(sum != 1){
            std::cerr << "The inputs given for partition " << l << " and state " << sO;
            std::cerr << " doesn't correspond to a probability distribution.\n";
            return; 
          }
        trans.set_transitions(l, sO, action, probs); 
        }
      }
   }
   

  // Reading the rewards of the sender 
  
  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);
  
  Srewards.init_rewards(states, A);
 
  double r;
  for(int l = 0; l < L-1; ++l){
    for(int s = 0; s < states[l]; ++s){
      for(int o = 0; o < A; ++o){
         TensorD v;
         for(int i = 0; i < A; ++i){
           inputFile >> r;
           if(r > 1 || r < 0){
             std::cerr << "The rewards are not in the required rank (inside [0,1]).\n";
             return; 
           } 
           v.push_back(r);
         }
         Srewards.set_rewards(l, s, o, v);
      }
    }
  }
  

  // Reading the rewards of the receiver 
  
  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);
  
  Rrewards.init_rewards(states, A);
 
  for(int l = 0; l < L-1; ++l){
    for(int s = 0; s < states[l]; ++s){
      for(int o = 0; o < A; ++o){
         TensorD v;
         for(int i = 0; i < A; ++i){
           inputFile >> r;
           if(r > 1 || r < 0){
             std::cerr << "The rewards are not in the required rank (inside [0,1]).\n";
             return; 
           } 
           v.push_back(r);
         }
         Rrewards.set_rewards(l, s, o, v);
      }
    }
  }
  

  // Reading the prior function, the probability of the outcomes given a state

  for(int i = 0; i < 3; ++i)
    std::getline(inputFile, textLine);

  mu.init_prior(states, A);
    
  double p;
  for(int l = 0; l < L; l++){
    int sMax = states[l];
    for(int s = 0; s < sMax; ++s){
      TensorD ps;
      double sum = 0; 
      for(int o = 0; o < A; ++o){
        inputFile >> p;
        sum += p;
        if(p > 1 || p < 0){
          std::cerr << "The prior function is incorrect\n";
          return;
        } 
        ps.push_back(p);
      }
      if(sum != 1){
          std::cerr << "The prior function is incorrect\n";
          return;
        } 
      mu.set_prior(l,s,ps);
    }
  }
  
  inputFile.close();

};



void print_enviroment(TensorI &states, size_t &A, transitions &trans, 
                     rewards<TypeReward::Sender> &Srewards, rewards<TypeReward::Receiver> &Rrewards, 
                     prior &mu){

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

  std::cout<< "The number of actions is : " << A << std::endl << std::endl;
   
  // Printing the probability transitions

  std::cout<< trans << std::endl;

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





