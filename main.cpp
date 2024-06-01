#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>

#include "Layer.h"
#include "Optimizer.h"
#include "DataHandler.h"
#include "Loss.h"
#include "NeuralNetwork.h"
#include "Utils.h"

using namespace NeuralNet;

int trainNN(NeuralNetwork &nn,
            CAnyLoss &loss,
            Optimizer &optimizer,
            DataHandler &trainHandler,
            DataHandler &evalHandler,
            int batchSize,
            int epochs,
            double lr,
            double lrDecay,
            const int numClasses)
{

    batchSize = 30;
    batchSize = 4;
    int runsInEpoch = trainHandler.getNumberOfSamples() / batchSize;
    auto evalBatch = evalHandler.getRandomBatch(5000);
    int scale = (numClasses == 10) ? 255 : 1;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        trainHandler.shuffleIndices();

        double lrCrt = lr / (1 + lrDecay * epoch);
        optimizer.lrUpdate(lrCrt);
        for (int batchRun = 0; batchRun < runsInEpoch; batchRun++)
        {
            auto batch = trainHandler.getRandomBatch(batchSize);
            Matrix prediction = nn.passForward(batch.features.transpose() / scale);
            Matrix targetVectors = labelsToOneHot(batch.labels, numClasses);
            Vector lossVector = loss->computeLoss(prediction, targetVectors);
            Matrix gradients = loss->computeGradient(prediction, targetVectors);
            Vector gradientMean = calculateRowwiseMean(gradients);
            nn.backprop(optimizer, gradientMean);

            int cnt = countCorrectPredictions(prediction, targetVectors);
            std::cout << "Epoch: " << epoch << " batchRun " << batchRun << '/'
                      << runsInEpoch << ", Loss: " << lossVector.transpose().mean()
                      << ", ratio: " << cnt << '/' << batchSize
                      << std::endl;
//            break;
        }


        Matrix predictionEval = nn.passForward(evalBatch.features.transpose() / scale);

        Matrix targetVectorsEval = labelsToOneHot(evalBatch.labels, numClasses);
        int cnt = countCorrectPredictions(predictionEval, targetVectorsEval);

        Matrix targetVectors = labelsToOneHot(evalBatch.labels, numClasses);
        Vector lossVector = loss->computeLoss(predictionEval, targetVectors);

        std::cout << "Epoch " << epoch << " ended." << std::endl;
        std::cout << "Eval dataset: " ", Loss: " << lossVector.transpose().mean()
                  << ", ratio: " << cnt << '/' << evalBatch.features.rows()
                  << std::endl;
//        break;
    }
    return 1;


}

int trainNew(NeuralNetwork &nn,
             CAnyLoss &loss,
             Optimizer &optimizer,
             DataHandler &trainHandler,
             DataHandler &evalHandler,
             int batchSize,
             int epochs,
             double lr,
             double lrDecay,
             const int numClasses)
{
    auto evalBatch = evalHandler.getRandomBatch(5000);
    int scale = (numClasses == 10) ? 255 : 1;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        trainHandler.shuffleIndices();

        double lrCrt = lr / (1 + lrDecay * epoch);
        optimizer.lrUpdate(lrCrt);
        nn.trainOnEpoch(loss, optimizer, trainHandler, epoch, batchSize, numClasses,
                        scale);


        auto res = nn.inferBatch(loss, evalBatch, numClasses, scale);

        std::cout << "Epoch #" << epoch << " ended." << " Loss: " << res.loss
                  << ", ratio: " << res.numberCorrect << '/'
                  << evalBatch.features.rows()
                  << std::endl;

    }

    return 1;
}

int main()
{
    std::string mnistTrain = "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_train.csv";
    std::string mnistTest = "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_test.csv";
    std::string SINGLETRAIN = "/Users/fuckingbell/programming/nn-from-scratch/data/sample_data.csv";
    DataHandler trainHandler;
    DataHandler evalHandler;
    trainHandler.readData(mnistTrain);
    evalHandler.readData(mnistTest);
    std::cout << "rows read by dataHandler: " << trainHandler.getNumberOfSamples()
              << std::endl;

    int NumClasses = 10;
    int pictureDim = 784;
    CAnyLayer layer1 = LinearLayer(pictureDim, 128);
    CAnyLayer activation1 = CwiseActivation::ReLu();
    CAnyLayer layer2 = LinearLayer(128, NumClasses);
    CAnyLayer activation2 = SoftmaxLayer();

    NeuralNetwork nn;
    nn.addLayer(std::move(layer1));
    nn.addLayer(std::move(activation1));
    nn.addLayer(std::move(layer2));
    nn.addLayer(std::move(activation2));

    CAnyLoss loss = MSELoss();

    int batchSize = 4;
    int epochs = 100;

    double lr = 0.3;
    double lrDecay = 0.5;
//    MomentumOptimizer optimizer(lr, 0.9);  // Learning rate and momentum
    double learningRate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    AMSGrad optimizer(learningRate, beta1, beta2, epsilon);
//    trainNN(nn, loss, optimizer, trainHandler, evalHandler, batchSize, epochs, lr,
//            lrDecay,
//            NumClasses);
    trainNew(nn, loss, optimizer,
             trainHandler,
             evalHandler,
             batchSize,
             epochs,
             lr,
             lrDecay, NumClasses
    );
    return 0;
}
