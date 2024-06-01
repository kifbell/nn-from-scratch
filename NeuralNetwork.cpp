//
// Created by Kirill Belyakov on 4/22/24.
//

#include "NeuralNetwork.h"


namespace NeuralNet
{


void NeuralNetwork::addLayer(CAnyLayer &&layer)
{
    layers_.push_back(std::move(layer));
}

Matrix NeuralNetwork::passForward(const Matrix &input)
{
    Matrix current_output = input;
    for (auto &layer: layers_)
    {
        current_output = layer->passForward(current_output);
    }
    return current_output;
}

void NeuralNetwork::backprop(Optimizer &optimizer, const Vector &output_gradient)
{
    Vector current_gradient = output_gradient;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
    {
//            std::cout << "current_gradient.transpose() " << current_gradient.transpose() << std::endl;
        current_gradient = (*it)->backprop(optimizer, current_gradient);
    }
}


void NeuralNetwork::trainOnEpoch(CAnyLoss &loss,
                                 Optimizer &optimizer,
                                 DataHandler &trainHandler,
                                 int epoch,
                                 int batchSize,
                                 const int numClasses,
                                 int scale
)
{
    trainHandler.shuffleIndices();


    int runsInEpoch = trainHandler.getNumberOfSamples() / batchSize;

    for (int batchRun = 0; batchRun < runsInEpoch; batchRun++)
    {
        auto batch = trainHandler.getRandomBatch(batchSize);
        Matrix prediction = passForward(batch.features.transpose() / scale);
        Matrix targetVectors = labelsToOneHot(batch.labels, numClasses);
        Vector lossVector = loss->computeLoss(prediction, targetVectors);
        Matrix gradients = loss->computeGradient(prediction, targetVectors);
        Vector gradientMean = calculateRowwiseMean(gradients);
        backprop(optimizer, gradientMean);

        int cnt = countCorrectPredictions(prediction, targetVectors);
        if (batchRun % 1000 ==0)
        {
            std::cout << "Epoch: " << epoch
                      << " progress " << (batchRun * 100) / runsInEpoch << '%'
                      << " batchRun " << batchRun << '/'
                      << runsInEpoch << ", Loss: " << lossVector.transpose().mean()
                      << ", ratio: " << cnt << '/' << batchSize
                      << std::endl;
        }
    }
}


InferenceResult NeuralNetwork::inferBatch(
        CAnyLoss &loss,
        DataBatch evalBatch,
        const int numClasses,
        int scale
)
{
    Matrix predictionEval = passForward(evalBatch.features.transpose() / scale);

    Matrix targetVectorsEval = labelsToOneHot(evalBatch.labels, numClasses);
    int cnt = countCorrectPredictions(predictionEval, targetVectorsEval);

    Matrix targetVectors = labelsToOneHot(evalBatch.labels, numClasses);
    Vector lossVector = loss->computeLoss(predictionEval, targetVectors);
    InferenceResult res(lossVector.transpose().mean(), cnt);
    return res;
}


}