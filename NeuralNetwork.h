//
// Created by Kirill Belyakov on 4/22/24.
//

#ifndef NN_FROM_SCRATCH_NEURALNETWORK_H
#define NN_FROM_SCRATCH_NEURALNETWORK_H

#include "eigen/Eigen/Core"
#include <iostream>
#include <memory>
#include <vector>

#include "DataHandler.h"
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Utils.h"


struct InferenceClassifierResult {
    double loss;
    int numberCorrect;

    InferenceClassifierResult(double l, int nc) : loss(l), numberCorrect(nc)
    {}
};

struct RegressorResult {
    double loss;
    RegressorResult(double l) : loss(l)
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

    void trainOnEpochClassifier(CAnyLoss &loss,
                                Optimizer &optimizer,
                                DataHandler &trainHandler,
                                int epoch,
                                int batchSize,
                                const int numClasses,
                                int scale);
    RegressorResult trainOnEpochRegressor(CAnyLoss &loss,
                                          Optimizer &optimizer,
                                          DataHandler &trainHandler,
                                          int epoch,
                                          int batchSize);


    InferenceClassifierResult inferBatchClassifier(
            CAnyLoss &loss,
            DataBatch evalBatch,
            const int numClasses,
            int scale);

    int trainClassifier(
            CAnyLoss &loss,
            Optimizer &optimizer,
            DataHandler &trainHandler,
            DataHandler &evalHandler,
            int batchSize,
            int epochs,
            double lr,
            double lrDecay,
            const int numClasses);

    int trainRegressor(
            CAnyLoss &loss,
            Optimizer &optimizer,
            DataHandler &trainHandler,
            int batchSize,
            int epochs,
            double lr,
            double lrDecay);

    static int runMNISTTest();

    static int runSinTest();

    static int runAllTests();
};
}// namespace NeuralNet

#endif//NN_FROM_SCRATCH_NEURALNETWORK_H
