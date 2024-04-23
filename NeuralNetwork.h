//
// Created by Kirill Belyakov on 4/22/24.
//

#ifndef NN_FROM_SCRATCH_NEURALNETWORK_H
#define NN_FROM_SCRATCH_NEURALNETWORK_H

#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>

#include "Layer.h"
#include "Optimizer.h"
#include "DataHandler.h"

namespace NeuralNet
{

class NeuralNetwork
{
private:
    std::vector<CAnyLayer> layers;

public:
    NeuralNetwork()
    {
    }

    void addLayer(CAnyLayer &layer)
    {
        layers.push_back(std::move(layer));
    }

    Vector passForward(const Vector &input)
    {
        Vector current_output = input;
        for (auto &layer: layers)
        {
            current_output = layer->passForward(current_output);
        }
        return current_output;
    }

    void backprop(Optimizer &optimizer, const Vector &output_gradient)
    {
        Vector current_gradient = output_gradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            current_gradient = (*it)->backprop(optimizer, current_gradient);
        }
    }
};
}

#endif //NN_FROM_SCRATCH_NEURALNETWORK_H
