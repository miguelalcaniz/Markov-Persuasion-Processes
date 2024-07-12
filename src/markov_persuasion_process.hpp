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
using TensorD = std::vector<double>;

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
    episode(size_t L_value): L(L_value){
        ep.resize(L_value);
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
    std::vector<SOA> ep;
};


// Declaration of the function that reads the values of the enviroment
void read_enviroment(size_t &L, TensorI &states, size_t &A, transitions &trans, 
                     rewards<TypeReward::Sender> &Srewards, rewards<TypeReward::Receiver> &Rrewards, 
                     prior &mu, const std::string& fileName);


// Declaration of the function that prints the values of the enviroment
void print_enviroment(TensorI &states, size_t &A, transitions &trans, 
                     rewards<TypeReward::Sender> &Srewards, rewards<TypeReward::Receiver> &Rrewards, 
                     prior &mu);


#endif /* MARKOV_PERSUASION_PROCESS_HPP */