#include <vector>
#include <iostream>
#include <cmath>
#include "../Matrix/matrix.h"


class NeuralNetwork
{
private:
    vector<Matrix> Layers;
    vector<Matrix> Weights;
    vector<Matrix> Bias;
    const float LearingRate = 0.2;


public:
    static float Activation(const float&) noexcept;
    static float ActivationDerivate(const float &) noexcept;

    explicit NeuralNetwork(const vector<int32_t>&) noexcept;

    vector<float> Predict(const vector<float>&) noexcept;
    void Fit(const vector<vector<float>> &, const vector<vector<float>> &, const  int32_t&);
    void PrintNeuralNetwork() noexcept;
};