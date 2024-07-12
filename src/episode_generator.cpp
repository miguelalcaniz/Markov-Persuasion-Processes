#include "episode_generator.hpp"
#include <vector>
#include <iostream>
#include <sstream>
#include <utility>
#include <map>


// Constructor which already sets the values
prior::prior(const TensorI &states_values, const size_t A_value)
{
  A = A_value;
  L = states_values.size();
  states = states_values;

  priorD.resize(L);
  for(int l = 0; l < L; l++)
    priorD[l].resize(states[l]);
}


// Method to initialize
void prior::init_prior(const TensorI &states_values, const size_t A_value)
{
  A = A_value;
  L = states_values.size();
  states = states_values;

  priorD.resize(L);
  for(int l = 0; l < L; l++)
    priorD[l].resize(states[l]);
}

// Method to set values of the prior
void prior::set_prior(const int l, const int s, const TensorD &probs)
{
  priorD[l][s] = probs;
  if(probs.size() != A) {
    std::cerr << "The vector of probabilities has the wrong size. And the size is " << probs.size() << std::endl;
    return;  
  }
}

// Method that generates a random outcome with the prior[l][s] discrete distribution
int prior::generate_outcome(const int l, const int s)
{
  std::random_device re;
  std::knuth_b knuth(re());
   
  TensorD &prob = priorD[l][s];
  std::discrete_distribution<> distribution(prob.begin(), prob.end());
  return distribution(knuth);
}

/// Definition of the stream operator for the transitions class

std::ostream &
operator<<(std::ostream &stream, transitions &trans)
{
   stream << "The probability transitions are:\n\n";
   size_t L = trans.L;
   size_t nStatesO;
   for(int l = 0; l < L-1; l++){
      nStatesO = trans.states[l];
      for(int sO = 0; sO < nStatesO; sO++){
        for(int action = 0; action < trans.A; action++){
         for(auto const& el: trans.get_transitions(l, sO, action))
           stream << el << ' ';
         stream << std::endl;
        }
      }
      stream << std::endl;
   }
  return stream;
}


// Constructor which already sets the values
transitions::transitions(const TensorI &state_values, const size_t A_value)
{
  A = A_value;
  L = state_values.size();
  states = state_values;
  tr.resize(L);
}

// Method for initializing the values of the variables
void transitions::init_transitions(const TensorI &state_values, const size_t A_value)
{
  A = A_value;
  L = state_values.size();
  states = state_values;
  tr.resize(L);
}

// Method that gives you the next state randomly with the given probability distribution
int transitions::next_state(const int l, const int origin, const int action)
{  
  std::random_device re;
  std::knuth_b knuth(re());
  
  TensorD &prob = tr[l][{origin, action}];
  std::discrete_distribution<> distribution(prob.begin(), prob.end());
  return distribution(knuth);
}
 

/// Definition of the stream operator for the rewards class
template<TypeReward R>
std::ostream &
operator<<(std::ostream &stream, rewards<R> &rewards){

  if(R == TypeReward::Sender)
     stream << "The sender rewards are:\n\n";
  if(R == TypeReward::Receiver)
     stream << "The receiver rewards are:\n\n";

  for(int l = 0; l < rewards.L-1; ++l){
   for(int s = 0; s < rewards.states[l]; ++s){
      for(int o = 0; o < rewards.A; ++o){
         for(int i = 0; i < rewards.A; ++i)
           stream << rewards.get_reward(l, s, o, i) << " ";
         stream << "\n";
      }
   }
   stream << std::endl;
  }
  return stream;
}


// Constructor which already sets de values
template<TypeReward R>
rewards<R>::rewards(const TensorI &states_values, const int A_value)
{
  A = A_value;
  L = states_values.size();
  states = states_values;
  rw = Tensor4D(states.size());
  for(int i = 0; i < states.size(); i++)
  rw[i] = Tensor3D(states[i], Tensor2D (A));
}

// Method for initializing the values
template<TypeReward R>
void rewards<R>::init_rewards(const TensorI &states_values, const int A_value)
{
  A = A_value;
  L = states_values.size();
  states = states_values;
  rw = Tensor4D(states.size());
  for(int i = 0; i < states.size(); i++)
  rw[i] = Tensor3D(states[i], Tensor2D (A));
}

template<TypeReward R>
void rewards<R>::set_rewards(const int l, const int state, const int outcome, 
                 const TensorD& rewards){
  if(l > L || state > states[l] || outcome > A)
  std::cerr << "The inputs given to the function are incorrect.\n";
  rw[l][state][outcome] = rewards;
}

template<TypeReward R>
const double& rewards<R>::get_reward(const int l, const int state, const int outcome, const int action) const
{
  if(l > L || state > states[l] || outcome > A || action > A)
    std::cerr << "The inputs given to the function are incorrect.\n";
  return rw[l][state][outcome][action];
}

//Method to obtain the values of the rewards
template<TypeReward R>
const TensorD& rewards<R>::get_rewards(const int l, const int state, const int outcome) const
{
  if(l > L || state > states[l] || outcome > A)
    std::cerr << "The inputs given to the function are incorrect.\n";
  return rw[l][state][outcome];
}


/// Definition of the stream operator for the prior class

std::ostream &
operator<<(std::ostream &stream, prior &mu){

  stream << "The prior function is:\n\n";

  for(int l = 0; l < mu.L; l++){
    int sMax = mu.states[l];
    for(int s = 0; s < sMax; ++s){
      for(int o = 0; o < mu.A; ++o)
        stream << mu.get_prior(l,s,o) << ' ';
      stream << std::endl;
    }
    stream << std::endl;
  }

  return stream;

}

// Explicit instance for the template for the types used

template rewards<TypeReward::Receiver>::rewards(const TensorI &states_values, const int A_value);
template rewards<TypeReward::Sender>::rewards(const TensorI &states_values, const int A_value);
template void rewards<TypeReward::Receiver>::init_rewards(const TensorI &states_values, const int A_value);
template void rewards<TypeReward::Sender>::init_rewards(const TensorI &states_values, const int A_value);
template void rewards<TypeReward::Receiver>::set_rewards(const int l, const int state, const int outcome, const TensorD& rewards);
template void rewards<TypeReward::Sender>::set_rewards(const int l, const int state, const int outcome, const TensorD& rewards);
template const double& rewards<TypeReward::Receiver>::get_reward(const int l, const int state, const int outcome, const int action) const;
template const double& rewards<TypeReward::Sender>::get_reward(const int l, const int state, const int outcome, const int action) const;
template const TensorD& rewards<TypeReward::Receiver>::get_rewards(const int l, const int state, const int outcome) const;
template const TensorD& rewards<TypeReward::Sender>::get_rewards(const int l, const int state, const int outcome) const;
template std::ostream &operator<< (std::ostream& stream, rewards<TypeReward::Sender>& R);
template std::ostream &operator<< (std::ostream& stream, rewards<TypeReward::Receiver>& R);
 
