//
// Created by Kirill Belyakov on 4/22/24.
//

#include "NeuralNetwork.h"

#include <iostream>
#include <memory>

#include "Layer.h"
#include "Optimizer.h"
#include "DataHandler.h"
#include "Loss.h"
#include "Utils.h"


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
        current_gradient = (*it)->backprop(optimizer, current_gradient);
    }
}


void NeuralNetwork::trainOnEpochClassifier(CAnyLoss &loss,
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
        if (batchRun % 1000 == 0)
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

RegressorResult NeuralNetwork::trainOnEpochRegressor(CAnyLoss &loss,
                                                     Optimizer &optimizer,
                                                     DataHandler &trainHandler,
                                                     int epoch,
                                                     int batchSize
)
{
    trainHandler.shuffleIndices();
    int runsInEpoch = trainHandler.getNumberOfSamples() / batchSize;


    double epochLoss = 0;
    for (int batchRun = 0; batchRun < runsInEpoch; batchRun++)
    {
        auto batch = trainHandler.getRandomBatch(batchSize);
        Matrix prediction = passForward(batch.features.transpose());
        Vector lossVector = loss->computeLoss(prediction, batch.labels.transpose());
        Matrix gradients = loss->computeGradient(prediction, batch.labels.transpose());
        Vector gradientMean = calculateRowwiseMean(gradients);
        backprop(optimizer, gradientMean);

        std::cout << "Epoch: " << epoch
                  << " progress " << (batchRun * 100) / runsInEpoch << '%'
                  << " batchRun " << batchRun << '/'
                  << runsInEpoch << ", Loss: " << lossVector.transpose().mean()
                  << std::endl;
        epochLoss += lossVector.transpose().mean();
    }

    RegressorResult res(epochLoss / runsInEpoch);
    return res;
}


InferenceClassifierResult NeuralNetwork::inferBatchClassifier(
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
    InferenceClassifierResult res(lossVector.transpose().mean(), cnt);
    return res;
}


int NeuralNetwork::trainClassifier(
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
        trainOnEpochClassifier(loss, optimizer, trainHandler, epoch, batchSize,
                               numClasses,
                               scale);

        auto res = inferBatchClassifier(loss, evalBatch, numClasses, scale);

        std::cout << "Epoch #" << epoch << " ended." << " Loss: " << res.loss
                  << ", ratio: " << res.numberCorrect << '/'
                  << evalBatch.features.rows()
                  << std::endl;
    }
    return 1;
}


int NeuralNetwork::trainRegressor(
        CAnyLoss &loss,
        Optimizer &optimizer,
        DataHandler &trainHandler,
        int batchSize,
        int epochs,
        double lr,
        double lrDecay)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        trainHandler.shuffleIndices();

        double lrCrt = lr / (1 + lrDecay * epoch);
        optimizer.lrUpdate(lrCrt);
        RegressorResult res = trainOnEpochRegressor(loss, optimizer, trainHandler,
                                                    epoch, batchSize);

        std::cout << "Epoch #" << epoch << " ended." << " Loss: " << res.loss
                  << std::endl;
    }
    return 1;
}

int NeuralNetwork::runMNISTTest()
{
    std::string mnistTrain = "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_train.csv";
    std::string mnistTest = "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_test.csv";
    DataHandler trainHandler;
    DataHandler evalHandler;
    trainHandler.readData(mnistTrain);
    evalHandler.readData(mnistTest);
    std::cout << "rows read by trainDataHandler: " << trainHandler.getNumberOfSamples()
              << std::endl;

    int NumClasses = 10;
    int pictureDim = 784;
    CAnyLayer layer1 = LinearLayer(pictureDim, 128, 1);
    CAnyLayer activation1 = CwiseActivation::ReLu();
    CAnyLayer layer2 = LinearLayer(128, NumClasses, 1);
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
    double learningRate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    AMSGrad optimizer(learningRate, beta1, beta2, epsilon);
    nn.trainClassifier(loss, optimizer,
                       trainHandler,
                       evalHandler,
                       batchSize,
                       epochs,
                       lr,
                       lrDecay, NumClasses
    );
    return 0;
}

int NeuralNetwork::runSinTest()
{
    std::string sinWaveTrain = "/Users/fuckingbell/programming/nn-from-scratch/data/sine_wave_train.csv";
    DataHandler trainHandler;
    trainHandler.readData(sinWaveTrain);
    std::cout << "rows read by trainDataHandler: " << trainHandler.getNumberOfSamples()
              << std::endl;

    int inputDim = 10;
    int innerDim = 100;
    double scale =100;
    CAnyLayer layer1 = LinearLayer(inputDim, innerDim, scale);
    CAnyLayer activation1 = CwiseActivation::ReLu();
    CAnyLayer layer2 = LinearLayer(innerDim, innerDim, scale);
    CAnyLayer activation2 = CwiseActivation::ReLu();
    CAnyLayer layer3 = LinearLayer(innerDim, 1, scale);

    NeuralNetwork nn;
    nn.addLayer(std::move(layer1));
    nn.addLayer(std::move(activation1));
    nn.addLayer(std::move(layer3));

    CAnyLoss loss = MSELoss();

    int batchSize = 1;
    int epochs = 100;
    double lrDecay = 1;

    double learningRate = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    AMSGrad optimizer(learningRate, beta1, beta2, epsilon);
//    MomentumOptimizer optimizer(learningRate, 0);
    nn.trainRegressor(loss,
                      optimizer,
                      trainHandler,
                      batchSize,
                      epochs,
                      learningRate,
                      lrDecay
    );



    double step = 0.1;
    Eigen::VectorXd X = Eigen::VectorXd::LinSpaced(100, 0, inputDim);

    Eigen::VectorXd real_sin = X.array().sin();

    int start = 10; // training history
    std::vector<Eigen::VectorXd> X_train;


    std::vector<double> X_new;
    std::vector<double> Y_new;
    X_new.push_back(X[X.size() - 1]);
    Y_new.push_back(real_sin[real_sin.size() - 1]);

    trainHandler.resetBatchIndex();
    DataBatch batch = trainHandler.getNextBatch(290);
    std::vector<double> y_train(batch.labels.data(),
                                batch.labels.data() + batch.labels.size());

    std::vector<double> X_in(y_train.end() - start, y_train.end());
    Eigen::VectorXd token = Eigen::Map<Eigen::VectorXd>(X_in.data(), X_in.size());

    for (int i = 0; i < 200; ++i)
    {
        X_new.push_back(X_new.back() + step);
        Eigen::VectorXd X_in_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
                X_in.data(), X_in.size());
        double next_y = nn.passForward(X_in_vec)(0, 0);
        Y_new.push_back(next_y);
        X_in.push_back(next_y);
        X_in.erase(X_in.begin());
    }

    std::cout << "Predicted values: " << std::endl;
    for (const auto& val : Y_new) {
        std::cout << val << std::endl;
    }
    return 1;
}

int NeuralNetwork::runAllTests()
{
    runMNISTTest();
//    runSinTest();
}
}