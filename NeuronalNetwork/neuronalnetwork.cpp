#include "neuronalnetwork.h"


using namespace std;

NeuralNetwork::NeuralNetwork(const vector<int32_t>& Topology) noexcept {
    for (int32_t i = 1; i < Topology.size(); ++i) {
        this->Bias.emplace_back(Matrix(Topology[i], 1));
        this->Weights.emplace_back(Matrix(Topology[i], Topology[i-1]));
        this->Layers.emplace_back(Matrix(0, 0));
    }
    this->Layers.emplace_back(Matrix(0, 0));
}

vector<float> NeuralNetwork::Predict(const vector<float>& Inputs) noexcept {
    this->Layers[0] = Matrix::FromArray(Inputs);
    for (int32_t i = 1; i < this->Layers.size(); ++i) {
        this->Layers[i] = Matrix::Multiply(
                this->Weights[(i - 1)],
                this->Layers[(i - 1)]
        );

        this->Layers[i] += this->Bias[(i - 1)];
        this->Layers[i].Map(NeuralNetwork::Activation);
    }

    return this->Layers[this->Layers.size() - 1].ToArray();
}


void NeuralNetwork::Fit(const vector<vector<float>>& M1, const vector<vector<float>>& M2, const int32_t& epochs) {

    for (int32_t _ = 0; _ < epochs; ++_) { //epocas
        for (int32_t i = 0; i < M1.size(); ++i) {
            auto outputs = Matrix::FromArray(this->Predict(M1[i]));

            auto target = Matrix::FromArray(M2[i]);
            auto output_errors = target - outputs;

            auto gradient = Matrix::Map(outputs, NeuralNetwork::ActivationDerivate);
            gradient *= output_errors;
            gradient *= this->LearingRate;

            auto hidden_T = Matrix::Transpose(this->Layers[(this->Layers.size() - 2)]);
            auto weiht_ho_deltas = Matrix::Multiply(gradient, hidden_T);

            this->Weights[this->Weights.size() - 1] += weiht_ho_deltas;
            this->Bias[this->Bias.size() - 1] += gradient;

            // back propagation
            for (auto j = (int32_t)(this->Weights.size() - 1); j > 0; --j) {
                auto who_ = Matrix::Transpose(this->Weights[j]);
                auto errors_ = Matrix::Multiply(who_, output_errors);


                auto gradien_ = Matrix::Map(this->Layers[j], NeuralNetwork::ActivationDerivate);
                gradien_ *= errors_;
                gradien_ *= this->LearingRate;

                auto input_ = Matrix::Transpose(this->Layers[(j - 1)]);
                auto weihts_deltas_ = Matrix::Multiply(gradien_, input_);

                this->Weights[(j - 1)] += weihts_deltas_;
                this->Bias[(j - 1)] += gradien_;
            }
        }
    }
}

void NeuralNetwork::PrintNeuralNetwork() noexcept {
    for (int32_t i = 0; i < this->Layers.size() - 1; ++i) {
        cout << "-------- Capa --------" << endl;
        this->Weights[i].Print();
        cout << "-------- Bias --------" << endl;
        this->Bias[i].Print();
    }
}

float NeuralNetwork::Activation(const float& val)  noexcept {
    return tanh(val);
}

float NeuralNetwork::ActivationDerivate(const float& val) noexcept {
    return (float)(1.0 - pow(NeuralNetwork::Activation(val), 2));
}