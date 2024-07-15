#ifndef EPISODE_GENERATOR_HPP
#define EPISODE_GENERATOR_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <map>
#include <sstream>
#include <random>

/* Alias vectors for making the code more readable */
using TensorI = std::vector<int>;
using Tensor2I = std::vector<TensorI>;
using Tensor3I = std::vector<Tensor2I>;
using Tensor4I = std::vector<Tensor3I>;
using TensorD = std::vector<double>;
using Tensor2D = std::vector<TensorD>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;


/* 
   Prior function class. Defining a probability distribution over
   outcomes at each state. We let μ(l, x) be the vector of probabilities
   with which outcomes ω ∈ Ω are sampled in state x ∈ X.
   (Inputs l and s determine the state of S)
   (Input w determine the outcome)
*/
    class prior {
    public:
        /* Default constructor used with init_prior to set values */
        prior() = default;
        
        /* Constructor that initializes the prior class with given state values and action size */
        prior(const TensorI &states_values, const size_t A_value);

        /* Method to initialize the prior with given state values and action size */
        void init_prior(const TensorI &states_values, const size_t A_value);

        /* Method to set the probability distribution for a specific state */
        void set_prior(const int l, const int s, const TensorD &probs);

        /* Method to get the probability distribution for a specific state */
        const TensorD& get_prior(const int l, const int s) const{
            return priorD[l][s];
        }

        /* Method to get the probability to get outcome (w) for a specific state (l,s) */
        const double& get_prior(const int l, const int s, const int w) const{
            return priorD[l][s][w];
        }

        /* Method to generate a random outcome based on the prior probability distribution for a given state */
        int generate_outcome(const int l, const int s);

       /* Stream operator to print the prior distribution */
       friend std::ostream &
       operator<<(std::ostream &stream, prior &mu);
 
    protected:
        size_t L; // Number of partitions of states
        size_t A; // Number of possible actions
        TensorI states; // Vector of states of every partition of S
        Tensor3D priorD; // Keeps the probability of having an outcome w in each state j of partition i (priorD[i][j][k])
    };

    /* 
       Transition function class. It sets the discrete probability distribution over
       states of the next partition l+1 for which next state will be x given a tuple 
       state-action (l, s, a) of partition l, state s and action a. 
       P (x′ | x, a) be the probability of moving from x ∈ X to x′ ∈ X by taking
       action a ∈ A.                                                                  
    */
    class transitions {
    public:
        /* Default constructor used with init_transitions to initialize class */
        transitions() = default;

        /* Constructor that initializes the transitions with given state values and action size */
        transitions(const TensorI &state_values, const size_t A_value);

        /* Method to initialize the transitions with given state values and action size */
        void init_transitions(const TensorI &state_values, const size_t A_value);

        /* Method to set the probability distribution for a specific state-action tuple over states*/
        void set_transitions(const int l,  const int s, const int a, const TensorD &probs){
            tr[l][s][a] = probs;
        }
         
        /* Method to set the probability for a specific state-action-action tuple */
        void set_transitions(const int l, const int s, const int a, 
                             const int x, const double p){
            tr[l][s][a][x] = p;
        }

        /* Method to get the probability for a specific state-action-action tuple */
        TensorD& get_transitions(const int l, const int s, const int a){
            return tr[l][s][a];
        }

        /* Method to generate a random next state based on the transition probabilities for a given state and action */
        int next_state(const int l, const int origin, const int action);
 
        /* Stream operator to print the transition function */
        friend std::ostream &
        operator<<(std::ostream &stream, transitions &trans);

    protected:
        size_t L; // Number of partitions of states
        size_t A; // Number of possible actions
        TensorI states; // Vector of states of every partition of S
        Tensor4D tr; // Given a state (partition and number of state (l, s)) and an action (o) you have the vector
                     // of probabilities to move to each state of the next partition
    };


    /* Enumerate class to differenciate from sender's reward and from receiver's reward */ 
    enum class TypeReward
    {
        Sender,
        Receiver
    };


    /*
      Rewards class. Computed as a template class to distinguish between Sender and Receiver.
      It sets the reward from every tuple state-outcome-action (l, s, a, o).
    */
    template<TypeReward R>
    class rewards {
    public:

        /* Default constructor used with init_transitions to initialize class */
        rewards() = default;

        /* Constructor that initializes the rewards class with given state values and action size */
        rewards(const TensorI &states_values, const int A_value);

        /* Method that initializes the rewards class with given state values and action size */
        void init_rewards(const TensorI &states_values, const int A_value);

        /* Method to set the reward value for a specific state, action, and outcome */
        void set_rewards(const int l, const int state, const int outcome, 
                         const TensorD& rewards);
        
        /* Method to get the rewards for a specific state and outcome */
        const TensorD& get_rewards(const int l, const int state, const int outcome) const;
        
        /* Method to get the reward value for a specific state, action, and outcome */
        const double& get_reward(const int l, const int state, const int outcome, const int action) const;

        /* Stream operator to print the rewards values */
        template<TypeReward TypeR>
        friend std::ostream &
        operator<<(std::ostream &stream, rewards<TypeR> &rewards);

    protected:
        size_t L; // Number of partitions of states
        size_t A; // Number of possible actions
        TensorI states; // Vector of states of every partition of S
        Tensor4D rw; // Tensor that keeps the rewards values for every state-action-outcome
    };


#endif /* EPISODE_GENERATOR_HPP */