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
#include "Loss.h"
#include "Utils.h"


struct InferenceResult
{
    double loss;
    int numberCorrect;

    InferenceResult(double l, int nc) : loss(l), numberCorrect(nc)
    {}

};
namespace NeuralNet
{

class NeuralNetwork
{
private:
    std::vector<CAnyLayer> layers_;

public:
    NeuralNetwork()
    {
    }

    void addLayer(CAnyLayer &&layer);

    Matrix passForward(const Matrix &input);

    void backprop(Optimizer &optimizer, const Vector &output_gradient);

    void trainOnEpoch(CAnyLoss &loss,
                      Optimizer &optimizer,
                      DataHandler &trainHandler,
                      int epoch,
                      int batchSize,
                      const int numClasses,
                      int scale
    );


    InferenceResult inferBatch(
            CAnyLoss &loss,
            DataBatch evalBatch,
            const int numClasses,
            int scale
    );
//
};
}

#endif //NN_FROM_SCRATCH_NEURALNETWORK_H
