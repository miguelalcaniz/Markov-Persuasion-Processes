#ifndef EPISODE_GENERATOR_HPP
#define EPISODE_GENERATOR_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <map>
#include <sstream>
#include <random>


    class prior {
    public:
        // Default constructor used with init_prior to set values
        prior() = default;
        
        // Constructor which already sets the values
        prior(const std::vector<int> &states_values, const int A_value);

        // Method to initialize
        void init_prior(const std::vector<int> &states_values, const int A_value);

        // Method to set values of the prior
        void set_prior(const int l, const int s, const std::vector<double> &probs);

        // Methods to obtain values of the prior
        const std::vector<double>& get_prior(const int l, const int s) const{
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
 

    private:
        // The probability of having an outcome k in each state j of partition i (prior[i][j][k])
        size_t L;
        size_t A;
        std::vector<int> states;
        std::vector<std::vector<std::vector<double>>> priorD;
    };

    class transitions {
    public:

        // Default constructor used with init_transitions to set values
        transitions() = default;

        // Constructor which already sets the values
        transitions(const std::vector<int> &state_values, const size_t A_value);

        // Method for initializing the values of the variables
        void init_transitions(const std::vector<int> &state_values, const size_t A_value);

        // Method for setting the values of the transition probabilities
        void set_transitions(const int l, const int origin, const int action, 
                             const std::vector<double> &probs){
            tr[l][{origin, action}] = probs;
        }

        // Method for obtaining the values of the transition probabilities
        std::vector<double>& get_transitions(const int l, const int origin, const int action){
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
        std::vector<int> states;
        // Given a state (partition and number of state), an outcome an an action you have the vector
        // of probabilities for each state of the next partition
        std::vector<std::map<std::pair<int, int>, std::vector<double>>> tr; 
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
        rewards(const std::vector<int> &states_values, const int A_value);

        // Method for initializing the sizes of the vectors
        void init_rewards(const std::vector<int> &states_values, const int A_value);

        // Method to set the values of the rewards
        void set_rewards(const int l, const int state, const int outcome, 
                         const std::vector<double>& rewards);
        
        //Method to obtain the values of the rewards
        const std::vector<double>& get_rewards(const int l, const int state, const int outcome) const;

        const double& get_reward(const int l, const int state, const int outcome, const int action) const;

       /// Stream operator.
       template<TypeReward TypeR>
       friend std::ostream &
       operator<<(std::ostream &stream, rewards<TypeR> &rewards);

    private:
        size_t A;
        size_t L;
        std::vector<int> states;
        std::vector<std::vector<std::vector<std::vector<double>>>> rw; 
    };


#endif /* EPISODE_GENERATOR_HPP */