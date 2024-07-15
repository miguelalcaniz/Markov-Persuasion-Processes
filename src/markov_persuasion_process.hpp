#ifndef MARKOV_PERSUASION_PROCESS_HPP
#define MARKOV_PERSUASION_PROCESS_HPP

#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include "episode_generator.hpp"
#include "OptOpt.hpp"

/* Alias vectors for making the code more readable */
using TensorI = std::vector<int>;
using Tensor2I = std::vector<TensorI>;
using Tensor3I = std::vector<Tensor2I>;
using Tensor4I = std::vector<Tensor3I>;
using TensorD = std::vector<double>;
using Tensor2D = std::vector<TensorD>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;


/* Enviroment variables to take all the information needed for the algorithms */
struct Enviroment {
    size_t L;
    TensorI states;
    size_t A;
    transitions trans;
    rewards<TypeReward::Sender> Srewards;
    rewards<TypeReward::Receiver> Rrewards; 
    prior mu;
};
 
/* Tuple class formed of a state, an outcome and an action */
class SOA {
public:
    /* Default constructor */ 
    SOA() = default;  

    /* Constructor that initializes the values */
    SOA(size_t xValue, size_t wValue, size_t aValue): x(xValue), w(wValue), a(aValue){}; 
    
    /* Methods for setting the state value */
    void setX(size_t value){
        x = value;
    };

    /* Methods for getting the state value */
    const size_t& getX() const{
        return x;
    };

    /* Methods for setting the outcome value */
    void setW(size_t value){
        w = value;
    };

    /*Methods for getting the outcome value*/
    const size_t& getW() const{
        return w;
    };

    /* Methods for setting the action value */
    void setA(size_t value){
        a = value;
    };

    /* Methods for getting the action value */
    const size_t& getA() const{
        return a;
    };
private:
    size_t x;  // State
    size_t w;  // Outcome
    size_t a;  // Action
};


/* Episode class that collects the tuples SOA of an episode */
class episode {
public:
    /* Constructor that sets the variable sizes */
    episode(TensorI values_states){
        L = values_states.size();
        states = values_states;
        ep.resize(L);

    };

    /* Methods for setting the SOA value */
    void set_soa(int n, SOA soa){
        ep[n] = soa;
    }

    /* Methods for getting the SOA value */
    const SOA& get_soa(int n) const{
        return ep[n];
    }

    /* Method that generates an episode with a given prior distribution, signaling scheme and probabilities of transitions */
    void generate_episode(prior mu, sign_scheme phi, transitions trans);

    /* Stream operator for the episode class */
    friend std::ostream &
    operator<<(std::ostream &stream, episode &ep);

private:
    size_t L; // Number of partitions of the states
    TensorI states; // Number of states in each partition
    std::vector<SOA> ep; // Vector that collects the tuples (s, o, a) of the episode
};

/* Derived class of the transitions for the estimation of the transition probabilities */
class est_transitions : public transitions {
public:
  /* Default constructor */
  est_transitions() = default;

  /* Constructor to initialize the estimated transitions with given state values and action size */
  est_transitions(const TensorI &state_values, const size_t A_value);

  /* Function that counts a new visited state-action-state tuple */
  void visited(const int l, const int s, const int a, const int x)
  {
    visits[l][s][a][x]++;
    out_of[l][s][a]++;
  };

  /* Method that updates the estimated transition function after a new episode is generated */
  void update_transitions(const episode &ep);

private:
  Tensor3I out_of; // Counts the state-action tulpes visited
  Tensor4I visits; // Counts the state-action-state tuples visited
};

/* Derived class of the prior for the estimation of the prior */
class est_prior : public prior {
public:
  /* Default constructor */
  est_prior() = default;

  /* Constructor to initialize the estimated prior with given state values and action size */
  est_prior(const TensorI &states_values, const size_t A_value);

  /* Function that counts a new visited state-outcome tuple */
  void visited(const int l, const int s, const int o){
    visits[l][s][o]++;
    out_of[l][s]++;
  }

  /* Method that updates the estimated prior function after a new episode is generated */
  void update_prior(const episode &ep);

private:
  Tensor2I out_of; // Counts the states visited
  Tensor3I visits; // Counts the states-outcomes tuples visited
};

/* Derived template class of the reward template class for the estimation of the reward */
/*
  As explained in the update rewards function, the rewards theorically are sampled from a 
  distribution with a mean value, so it makes sense to estimate it though the mean of several
  samples.
  But in our code we just take a fixed value for the rewards given in the data.txt file, so
  doing the mean of the same values is unnecessary. Anyway the estimation of the reward 
  calculating the mean of the seen values is programmed.
*/
template<TypeReward R>
class est_rewards : public rewards<R> {
public:
  /* Default constructor */
  est_rewards() = default;

  /* Constructor to initialize the estimated rewards with given state values and action size */
  est_rewards(const TensorI &states_values, const int A_value);
     
  /* Function that counts a new visited state-outcome-action tuple */
  void visited(const int l, const int s, const int o, const int a){
    visits[l][s][o][a]++;
  }

  /* Method that updates the estimated rewards values after a new episode is generated */
  void update_rewards(const episode &ep, const rewards<R> &rw);

private:
  Tensor4I visits; // Counts the visits for the state-outcome-action tuples
};

/* Structure that takes all the estimated function and values by the sender */
struct Estimators {
    est_prior estimated_mu;
    est_transitions estimated_trans;
    est_rewards<TypeReward::Sender> estimated_SR;
    est_rewards<TypeReward::Receiver> estimated_RR;
};

/* Function that reads the information of the enviroment from a .txt file */
void read_enviroment(Enviroment &env, const std::string& fileName);

/* Function that prints the information about the enviroment */
void print_enviroment(Enviroment &env);

/* Algorithm 1 (Sender-Receivers Interaction) */
episode S_R_interaction(Enviroment& env, sign_scheme phi);

/* Algorithm 2 OPPS (Optimistic Persuasive Policy Search) */
Estimators OPPS(Enviroment& env, unsigned int T);

/* Function that prints the Estimators */
void print_estimators(Estimators& est);

#endif /* MARKOV_PERSUASION_PROCESS_HPP */