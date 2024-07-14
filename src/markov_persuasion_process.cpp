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
    int outcome, action;
     
    for(int l = 0; l < L-1; ++l){
      outcome = mu.generate_outcome(l, actual_state);
      action = phi.recommendation(l, actual_state, outcome);
      SOA soa(l, outcome, action);
      ep[l] = soa;
      actual_state = trans.next_state(l, actual_state, action);
    }
}


// This function reads a .txt file and sets all the information of the enviroment into the variables 

void read_enviroment(Enviroment &env, const std::string& fileName)
{
  // Defining some references to the variables to avoid writing env.
  size_t& L = env.L;
  TensorI& states = env.states;
  size_t& A = env.A;
  transitions& trans = env.trans;
  rewards<TypeReward::Sender>& Srewards = env.Srewards;
  rewards<TypeReward::Receiver>& Rrewards = env.Rrewards; 
  prior& mu = env.mu;

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
  for(int l = 0; l < L; ++l){
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
 
  for(int l = 0; l < L; ++l){
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



void print_enviroment(Enviroment& env)
{
  // Defining some references to the variables to avoid writing env.
  size_t& L = env.L;
  TensorI& states = env.states;
  size_t& A = env.A;
  transitions& trans = env.trans;
  rewards<TypeReward::Sender>& Srewards = env.Srewards;
  rewards<TypeReward::Receiver>& Rrewards = env.Rrewards; 
  prior& mu = env.mu;

  std::cout<< "--------------------------------------" << std::endl;
  std::cout<< "TESTING THE CLASS EPISODE" << std::endl << std::endl;

  // Printing the states

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
  for(int l = 0; l < ep.L; ++l){
    stream << '(' << ep.get_soa(l).getX() << ", " << ep.get_soa(l).getW() << ", ";
    stream << ep.get_soa(l).getA() << ')';
    if(l != ep.L-1) std::cout<< " --> ";
  }
  stream << std::endl;
  return stream;
};


est_transitions::est_transitions(const TensorI &state_values, const size_t A_value){
  
  A = A_value; 
  L = state_values.size();
  states = state_values;
  tr.resize(L-1);

  visits.resize(L-1);
  out_of.resize(L-1);
  for(int l = 0; l < L-1; ++l){
    tr[l].resize(states[l]);
    visits[l].resize(states[l]);
    out_of[l].resize(states[l]);
    for(int s = 0; s < states[l]; ++s){
      tr[l][s] = Tensor2D(A, TensorD(states[l+1]));
      visits[l][s].resize(A);
      out_of[l][s] = TensorI(A, 0);
      for(int a = 0; a < A; ++a)
        visits[l][s][a] = TensorI(states[l+1], 0);
    }
  }
};

void est_transitions::update_transitions(const episode &ep){

  for(int l = 0; l < L-1; ++l){
    const SOA& origin = ep.get_soa(l);
    const int& s = origin.getX();
    const int& o = origin.getW(); 
    const int& a = origin.getA();
    const int& x = ep.get_soa(l).getX(); 
    TensorD probs(states[l+1]);
    visited(l, s, a, x);
    for(int x = 0; x < states[l+1]; ++x){
      probs[x] = static_cast<double>(visits[l][s][a][x]) / out_of[l][s][a];
    }
    set_transitions(l, s, a, probs);
  }
};


est_prior::est_prior(const TensorI &states_values, const size_t A_value){
  A = A_value;
  L = states_values.size();
  states = states_values;

  priorD.resize(L);
  out_of.resize(L);
  visits.resize(L);
  for(int l = 0; l < L; l++){
    priorD[l].resize(states[l]);
    out_of[l] = TensorI(states[l], 0);
    visits[l] = Tensor2I(states[l], TensorI(A, 0));
  }
}

void est_prior::update_prior(const episode &ep){
  for(int l = 0; l < L; ++l){
    const SOA& origin = ep.get_soa(l);
    const int& s = origin.getX();
    const int& o = origin.getW();
    visited(l, s, o);
    TensorD probs(A);
    for(int o = 0; o < A; ++o)
      probs[o] = static_cast<double>(visits[l][s][o]) / out_of[l][s];
    set_prior(l,s,probs);
  }
}


// Constructor which already sets de values
template<TypeReward R>
est_rewards<R>::est_rewards(const TensorI &states_values, const int A_value)
{
  this-> A = A_value;
  this-> L = states_values.size();
  this-> states = states_values;
  this-> rw = Tensor4D(this->L);
  visits.resize(this->L);
  for(int s = 0; s < this->L; ++s){
    this->rw[s] = Tensor3D(this->states[s], Tensor2D (this->A, TensorD(this->A)));
    visits[s] = Tensor3I(this->states[s], Tensor2I (this->A, TensorI(this->A, 0)));
  };
}

template<TypeReward R>
void est_rewards<R>::update_rewards(const episode &ep, const rewards<R> &rr)
{
  for(int l = 0; l < this->L; ++l){
    const SOA& origin = ep.get_soa(l);
    const int& s = origin.getX();
    const int& o = origin.getW();
    const int& a = origin.getA();
    // Update the estimated rewards, theorically the reward corresponds to a prob distribution
    // Now we are just updating the estimated reward with the mean of all the seen rewards 
    // (that correspond to a fix value so it doesn't make much sense)
    this->rw[l][s][o][a] = rr.get_reward(l, s, o, a);
  }
}


// Algorithm 1 (Sender-Receivers Interaction at episode t)

episode S_R_interaction(Enviroment& env, sign_scheme phi){ 
    
  // Defining some references to the variables to avoid writing env.
  size_t& L = env.L;
  TensorI& states = env.states;
  size_t& A = env.A;
  transitions& trans = env.trans;
  rewards<TypeReward::Sender>& Srewards = env.Srewards;
  rewards<TypeReward::Receiver>& Rrewards = env.Rrewards; 
  prior& mu = env.mu;

  // Declaration of the auxiliar variables for the algorithm
  int action, outcome, actual_state = 0;
  
  episode ep(states);

  for(int l = 0; l < L; ++l){
    outcome = mu.generate_outcome(l, actual_state);
    action = phi.recommendation(l, actual_state, outcome);
    SOA soa(actual_state, outcome, action);
    ep.set_soa(l, soa);
    if(l != L-1)
      actual_state = trans.next_state(l, actual_state, action);
  }
  
  return ep;
}


// Algorithm 2 OPPS

Estimators OPPS(Enviroment& env, unsigned int T){

  // Defining some references to the variables to avoid writing env.
  size_t& L = env.L;
  TensorI& states = env.states;
  size_t& A = env.A;
  rewards<TypeReward::Sender>& Srewards = env.Srewards;
  rewards<TypeReward::Receiver>& Rrewards = env.Rrewards; 
  prior& mu = env.mu;

  // Initializing the estimators variables
  est_prior estimated_mu(states, A);
  est_transitions estimated_trans(states, A);
  est_rewards<TypeReward::Sender> estimated_SR(states, A);
  est_rewards<TypeReward::Receiver> estimated_RR(states, A);

  sign_scheme phi;
  phi.init_scheme(states, A);

  for(int t = 0; t < T; ++t){
    OptOpt(phi);
    episode ep = S_R_interaction(env, phi);
    // Update of the estimated prior
    estimated_mu.update_prior(ep);
    // Update estimation of the transitions function 
    estimated_trans.update_transitions(ep);
    // Update rewards
    estimated_SR.update_rewards(ep, Srewards);
    estimated_RR.update_rewards(ep, Rrewards);
  }

  Estimators estimators; 
  estimators.estimated_mu = estimated_mu; 
  estimators.estimated_trans = estimated_trans;
  estimators.estimated_SR = estimated_SR;
  estimators.estimated_RR = estimated_RR;
  return estimators;
}


void print_estimators(Estimators& est)
{
  // Defining some references to the variables to avoid writing env.
  est_prior& estimated_mu = est.estimated_mu;
  est_transitions& estimated_trans = est.estimated_trans;
  est_rewards<TypeReward::Sender>& estimated_SR = est.estimated_SR;
  est_rewards<TypeReward::Receiver>& estimated_RR = est.estimated_RR;
 
 // Printing the probability transitions estimation

  std::cout<< estimated_trans << std::endl;

  // Printing the sender rewards

  std::cout<< estimated_SR << std::endl;

  // Printing the receiver rewards

  std::cout<< estimated_RR << std::endl;

  std::cout<< std::endl;

  // Printing the prior function

  std::cout<< estimated_mu << std::endl;
}


template est_rewards<TypeReward::Receiver>::est_rewards(const TensorI &states_values, const int A_value);
template est_rewards<TypeReward::Sender>::est_rewards(const TensorI &states_values, const int A_value);
template void est_rewards<TypeReward::Receiver>::update_rewards(const episode &ep, const rewards<TypeReward::Receiver> &rr);
template void est_rewards<TypeReward::Sender>::update_rewards(const episode &ep, const rewards<TypeReward::Sender> &rr);
 




