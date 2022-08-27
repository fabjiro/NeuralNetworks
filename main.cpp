#include <iostream>
#include <random>
#include <time.h>
#include "NeuronalNetwork/neuronalnetwork.h"

template <typename T>
ostream& operator<<(ostream& os, const std::vector<T>& V){
    for(const auto& item: V)
        std::cout << item;
    std::cout << std::endl;
    return os;
}

int32_t main() {
    srand(time(0));

    std::vector<std::vector<float>> inputs = {
            {0,1},
            {1,0},
            {0,0},
            {1,1}
    };

    std::vector<std::vector<float>> outputs = {
            {0},
            {0},
            {1},
            {1}
    };

    auto brain = NeuralNetwork({2,3,1});

    //brain.PrintNeuralNetwork();
    brain.Fit(inputs, outputs, 5000);

    //brain.PrintNeuralNetwork();

    std::cout << brain.Predict(inputs[0]);
    std::cout << brain.Predict(inputs[1]);
    std::cout << brain.Predict(inputs[2]);
    std::cout << brain.Predict(inputs[3]);
    return EXIT_SUCCESS;
}
