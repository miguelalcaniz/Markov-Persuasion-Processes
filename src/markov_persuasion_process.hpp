#ifndef MARKOV_PERSUASION_PROCESS_HPP
#define MARKOV_PERSUASION_PROCESS_HPP

#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include "episode_generator.hpp"
#include "OptOpt.hpp"

// Alias for making the code more readable
using TensorI = std::vector<int>;
using Tensor2I = std::vector<TensorI>;
using Tensor3I = std::vector<Tensor2I>;
using Tensor4I = std::vector<Tensor3I>;
using TensorD = std::vector<double>;
using Tensor2D = std::vector<TensorD>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;


// Definition of the struct Settings
struct Enviroment {
    size_t L;
    TensorI states;
    size_t A;
    transitions trans;
    rewards<TypeReward::Sender> Srewards;
    rewards<TypeReward::Receiver> Rrewards; 
    prior mu;
};
 
// State-Outcome-Action
class SOA {
public:
    // Constructors
    
    // Default constructor 
    SOA() = default;  

    // Constructor with parameters
    SOA(size_t xValue, size_t wValue, size_t aValue): x(xValue), w(wValue), a(aValue){}; 
    
    // Methods for setting and getting the values
    void setX(size_t value){
        x = value;
    };

    const size_t& getX() const{
        return x;
    };

    void setW(size_t value){
        w = value;
    };

    const size_t& getW() const{
        return w;
    };

    void setA(size_t value){
        a = value;
    };

    const size_t& getA() const{
        return a;
    };
    
private:
    size_t x;  // State
    size_t w;  // Outcome
    size_t a;  // Action
};


class episode {
public:
    // Constructor
    episode(TensorI values_states){
        L = values_states.size();
        states = values_states;
        ep.resize(L);

    };

    void set_soa(int n, SOA soa){
        ep[n] = soa;
    }

    const SOA& get_soa(int n) const{
        return ep[n];
    }

    // Method that generates an episode with a given prior distribution, siganling scheme and probabilities of transitions
    void generate_episode(prior mu, sign_scheme phi, transitions trans);

    /// Stream operator.
    friend std::ostream &
    operator<<(std::ostream &stream, episode &ep);

private:
    size_t L;
    TensorI states;
    std::vector<SOA> ep;
};

/*
class TupleVisited{
public:
  // Constructor
  TupleVisited(Enviroment& env);

  void visited(const int l, const int s, const int a, const int x)
  {
    visits[l][s][a][x]++;
    out_of[l][s][a]++;
  };

  // Method that updates the estimated transition function
  void update_transitions(const episode &ep, transitions &trans_est);

private:
  size_t L;
  size_t A;
  TensorI states;
  Tensor3I out_of;
  Tensor4I visits;
};
*/

class est_transitions : public transitions {
public:
  est_transitions(const TensorI &state_values, const size_t A_value);


  void visited(const int l, const int s, const int a, const int x)
  {
    visits[l][s][a][x]++;
    out_of[l][s][a]++;
  };

  // Method that updates the estimated transition function
  void update_transitions(const episode &ep);

private:
  Tensor3I out_of;
  Tensor4I visits;
};

class est_prior : public prior {
public:
  est_prior() = default;

  est_prior(const TensorI &states_values, const size_t A_value);

  void visited(const int l, const int s, const int o){
    visits[l][s][o]++;
    out_of[l][s]++;
  }

  void update_prior(const episode &ep);

private:
  Tensor2I out_of;
  Tensor3I visits;
};

template<TypeReward R>
class est_rewards : public rewards<R> {
public:
  est_rewards() = default;

  est_rewards(const TensorI &states_values, const int A_value);
     
  void visited(const int l, const int s, const int o, const int a){
    visits[l][s][o][a]++;
  }

  void update_rewards(const episode &ep, const rewards<R> &rw);

private:
  Tensor4I visits;
};

struct Estimators {
    est_prior estimated_mu;
    transitions estimated_trans;
    est_rewards<TypeReward::Sender> estimated_SR;
    est_rewards<TypeReward::Receiver> estimated_RR;
};

// Declaration of the function that reads the values of the enviroment
void read_enviroment(Enviroment &env, const std::string& fileName);


// Declaration of the function that prints the values of the enviroment
void print_enviroment(Enviroment &env);


// Algorithm 1 (Sender-Receivers Interaction at episode t)

episode S_R_interaction(Enviroment& env, sign_scheme phi);


// Algorithm 2 OPPS

Estimators OPPS(Enviroment& env, unsigned int T);

#endif /* MARKOV_PERSUASION_PROCESS_HPP */