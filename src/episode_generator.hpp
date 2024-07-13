#ifndef EPISODE_GENERATOR_HPP
#define EPISODE_GENERATOR_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <map>
#include <sstream>
#include <random>

// Alias for making the code more readable
using TensorI = std::vector<int>;
using Tensor2I = std::vector<TensorI>;
using Tensor3I = std::vector<Tensor2I>;
using Tensor4I = std::vector<Tensor3I>;
using TensorD = std::vector<double>;
using Tensor2D = std::vector<TensorD>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;


    class prior {
    public:
        // Default constructor used with init_prior to set values
        prior() = default;
        
        // Constructor which already sets the values
        prior(const TensorI &states_values, const size_t A_value);

        // Method to initialize
        void init_prior(const TensorI &states_values, const size_t A_value);

        // Method to set values of the prior
        void set_prior(const int l, const int s, const TensorD &probs);

        // Methods to obtain values of the prior
        const TensorD& get_prior(const int l, const int s) const{
            return priorD[l][s];
        }

        const double& get_prior(const int l, const int s, const int w) const{
            return priorD[l][s][w];
        }

        // Method that generates a random outcome with the prior[l][s] discrete distribution
        int generate_outcome(const int l, const int s);

       /// Stream operator.
       friend std::ostream &
       operator<<(std::ostream &stream, prior &mu);
 

    protected:
        size_t L;
        size_t A;
        TensorI states;
        // The probability of having an outcome k in each state j of partition i (priorD[i][j][k])
        Tensor3D priorD;
    };


    class transitions {
    public:

        // Default constructor used with init_transitions to set values
        transitions() = default;

        // Constructor which already sets the values
        transitions(const TensorI &state_values, const size_t A_value);

        // Method for initializing the values of the variables
        void init_transitions(const TensorI &state_values, const size_t A_value);

        // Methods for setting the values of the transition probabilities
        void set_transitions(const int l, const int origin, const int action, 
                             const TensorD &probs){
            tr[l][{origin, action}] = probs;
        }
         
        void set_transitions(const int l, const int origin, const int action, 
                             const int destination, const double p){
            tr[l][{origin, action}][destination] = p;
        }

        // Method for obtaining the values of the transition probabilities
        TensorD& get_transitions(const int l, const int origin, const int action){
            return tr[l][{origin, action}];
        }

        // Method that gives you the next state randomly with the given probability distribution
        int next_state(const int l, const int origin, const int action);
 
       /// Stream operator.
       friend std::ostream &
       operator<<(std::ostream &stream, transitions &trans);

    private:
        size_t L;
        size_t A; 
        TensorI states;
        // Given a state (partition and number of state), an outcome an an action you have the vector
        // of probabilities for each state of the next partition
        std::vector<std::map<std::pair<int, int>, TensorD>> tr; 
    };

    // Class to differenciate reward from sender and from receiver
    enum class TypeReward
    {
        Sender,
        Receiver
    };

    template<TypeReward R>
    class rewards {
    public:

        // Default constructor, used with init reward to set the values
        rewards() = default;

        // Constructor which already initializes the sizes of the vectors
        rewards(const TensorI &states_values, const int A_value);

        // Method for initializing the sizes of the vectors
        void init_rewards(const TensorI &states_values, const int A_value);

        // Method to set the values of the rewards
        void set_rewards(const int l, const int state, const int outcome, 
                         const TensorD& rewards);
        
        //Method to obtain the values of the rewards
        const TensorD& get_rewards(const int l, const int state, const int outcome) const;

        const double& get_reward(const int l, const int state, const int outcome, const int action) const;

       /// Stream operator.
       template<TypeReward TypeR>
       friend std::ostream &
       operator<<(std::ostream &stream, rewards<TypeR> &rewards);

    private:
        size_t A;
        size_t L;
        TensorI states;
        Tensor4D rw; 
    };


#endif /* EPISODE_GENERATOR_HPP */