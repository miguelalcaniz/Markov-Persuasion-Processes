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
        // Constructor
        prior(){};

        // Method to initialize
        void init_prior(const std::vector<int> &states_values, const int w_value){
          w = w_value;
          L = states_values.size();
          states = states_values;

          priorD.resize(L);
          for(int l = 0; l < L; l++)
            priorD[l].resize(states[l]);
        }

        // Methods to set an obtain values of the prior
        void set_prior(const int l, const int s, const std::vector<double> &probs){
            priorD[l][s] = probs;
            if(probs.size() != w) {
              std::cerr << "The vector of probabilities has the wrong size. And the size is " << probs.size() << std::endl;
              return;
            }
        }

        std::vector<double> get_prior(const int l, const int s){
            return priorD[l][s];
        }

        double get_prior(const int l, const int s, const int w){
            return priorD[l][s][w];
        }

        int generate_outcome(const int l, const int s){
          
          std::random_device re;
          std::knuth_b knuth(re());
   
          std::vector<double> &prob = priorD[l][s];
          std::discrete_distribution<> distribution(prob.begin(), prob.end());
          return distribution(knuth);
          
        }

       /// Stream operator.
       friend std::ostream &
       operator<<(std::ostream &stream, prior &mu);
 

    private:
        // The probability of having an outcome k in each state j of partition i (prior[i][j][k])
        size_t L;
        size_t w;
        std::vector<int> states;
        std::vector<std::vector<std::vector<double>>> priorD;
    };

    class transitions {
    public:

        // Constructor
        transitions(){};

        // Method for setting the value L
        void setL(size_t value){
            L = value;
        }

        //Method for initializing the vector with size L
        void init_transitions(std::vector<int> &state_values, size_t a_value){
            a = a_value;
            L = state_values.size();
            states = state_values;
            tr.resize(L);
        }

        void set_transitions(int l, int origin, int action, std::vector<double> probs){
            tr[l][{origin, action}] = probs;
        }

        std::vector<double> get_transitions(int l, int origin, int action){
            return tr[l][{origin, action}];
        }

        int next_state(int l, int origin, int action){
          
          std::random_device re;
          std::knuth_b knuth(re());
   
          std::vector<double> &prob = tr[l][{origin, action}];
          std::discrete_distribution<> distribution(prob.begin(), prob.end());
          return distribution(knuth);
          
        }
 
       /// Stream operator.
       friend std::ostream &
       operator<<(std::ostream &stream, transitions &trans);

    private:
        size_t L;
        size_t a; 
        std::vector<int> states;
        // Given a state (partition and number of state), an outcome an an action you have the vector
        // of probabilities for each state of the next partition
        std::vector<std::map<std::pair<int, int>, std::vector<double>>> tr; 
    };

    class rewards {
    public:

        // Constructor
        rewards(){};

        //Method for initializing the vector with size L
        void init_rewards(std::vector<int> &states_values, int w_value, int a_value){
            a = a_value;
            w = w_value;
            L = states_values.size();
            states = states_values;
            R = std::vector<std::vector<std::vector<std::vector<double>>>>(states.size());
            for(int i = 0; i < states.size(); i++)
                R[i] = std::vector<std::vector<std::vector<double>>>(states[i], std::vector<std::vector<double>> (w));
        }

        void set_rewards(int l, int state, int outcome, std::vector<double>& rewards){
            R[l][state][outcome] = rewards;
        }

        std::vector<double> get_rewards(int l, int state, int outcome){
            return R[l][state][outcome];
        }

        double get_reward(int l, int state, int outcome, int action){
            return R[l][state][outcome][action];
        }

       /// Stream operator.
       friend std::ostream &
       operator<<(std::ostream &stream, rewards &rewards);

    private:
        size_t a;
        size_t w;
        size_t L;
        std::vector<int> states;
        std::vector<std::vector<std::vector<std::vector<double>>>> R; 
    };

/* EJEMPLO DE TEMPLATE PARA HACER REWARDS SENDER Y REWARDS RECEIVER


#include <iostream>
#include <type_traits>

enum class Sign {
    Positive,
    Negative
};

template <Sign S>
class Number {
public:
    Number(int value) {
        if constexpr (S == Sign::Positive) {
            value_ = (value < 0) ? -value : value;  // Ensure the value is positive
        } else if constexpr (S == Sign::Negative) {
            value_ = (value > 0) ? -value : value;  // Ensure the value is negative
        }
    }

    void print() const {
        if constexpr (S == Sign::Positive) {
            std::cout << "Positive Number: " << value_ << std::endl;
        } else if constexpr (S == Sign::Negative) {
            std::cout << "Negative Number: " << value_ << std::endl;
        }
    }

private:
    int value_;
};

int main() {
    Number<Sign::Positive> posNumber(10);
    Number<Sign::Negative> negNumber(-10);

    posNumber.print();  // Output: Positive Number: 10
    negNumber.print();  // Output: Negative Number: -10

    return 0;
}


*/
#endif /* EPISODE_GENERATOR_HPP */