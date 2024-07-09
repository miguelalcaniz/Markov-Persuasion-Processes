#ifndef MARKOV_PERSUASION_PROCESS_HPP
#define MARKOV_PERSUASION_PROCESS_HPP

#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include "episode_generator.hpp"
#include "OptOpt.hpp"


// State-Outcome-Action
class SOA {
public:
    // Constructors
    
    // Default constructor 
    SOA(){};  

    // Constructor with parameters
    SOA(size_t xValue, size_t wValue, size_t aValue): x(xValue), w(wValue), a(aValue){}; 
    
    // Methods for setting and getting the values
    void setX(size_t value){
        x = value;
    };

    size_t getX() const {
        return x;
    };

    void setW(size_t value){
        w = value;
    };

    size_t getW() const {
        return w;
    };

    void setA(size_t value){
        a = value;
    };

    size_t getA() const {
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

    SOA get_soa(int n){
        return ep[n];
    }


    void generate_episode(prior mu, sign_scheme phi, transitions trans){

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
    /// Stream operator.
    friend std::ostream &
    operator<<(std::ostream &stream, episode &ep);



private:
    size_t L;
    std::vector<SOA> ep;
};


void read_enviroment(size_t &L, std::vector<int> &states, size_t &a, transitions &trans, 
                     rewards &Srewards, rewards &Rrewards, size_t &w, prior &mu, const std::string& fileName);


void print_enviroment(std::vector<int> &states, size_t &a, transitions &trans, 
                     rewards &Srewards, rewards &Rrewards, size_t &w, prior &mu);


#endif /* MARKOV_PERSUASION_PROCESS_HPP */