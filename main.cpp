#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>

#include "Layer.h"
#include "Optimizer.h"
#include "DataHandler.h"
#include "Loss.h"
#include "NeuralNetwork.h"

using namespace NeuralNet;


int argmax(const Vector &vec)
{
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return maxIndex;
}


Vector labelToOneHot(int label)
{
    Vector oneHot = Vector::Zero(
            10);
    if (label >= 0 && label < 10)
    {
        oneHot(label) = 1.0;
    } else
    {
        std::cerr << "Label out of bounds"
                  << std::endl;
    }
    return oneHot;
}

double calculateMean(const std::vector<double> &numbers)
{
    if (numbers.empty()) return 0.0;  // Return 0 if the vector is empty to avoid division by zero

    double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    return sum / numbers.size();
}

Vector calculateColwiseMean(const Eigen::MatrixXd &matrix)
{
    if (matrix.rows() == 0)
        return Vector();  // Return an empty vector if the matrix is empty

    // Compute the mean of each column
    Vector mean = matrix.colwise().mean();

    return mean;
}


int trainNN(NeuralNetwork &nn,
            MSELoss &loss,
            Optimizer &optimizer,
            DataHandler &trainHandler,
            int batchSize,
            int epochs
)
{


    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        auto batch = trainHandler.getRandomBatch(batchSize);


        Matrix gradients(batchSize, 10);
        std::vector<double> losses;
        for (size_t idx = 0; idx < batchSize; idx++)
        {
//            Vector input(batch.features.row(idx).size());
            Vector input = batch.features.row(idx);
            Vector target = labelToOneHot(batch.labels[idx]);

//            std::cout << "input" << std::endl;
//            std::cout << batch.features.row(idx).transpose() << std::endl;

            Vector prediction = nn.passForward(input);

//            std::cout << "target: " << batch.first[idx] << ", prediction: " << argmax(prediction)<< std::endl;
//            std::cout << "prediction: " << prediction << std::endl;
//            std::cout << "    target: " << target << std::endl;

            double current_loss = loss.compute_loss(prediction, target);
//            std::cout << "current_loss: " <<current_loss << std::endl;
            Vector gradient = loss.compute_gradient(prediction, target);

            // batch loss history
            gradients.row(idx) = gradient;
            losses.push_back(current_loss);
        }

        // step by mean batch gradient
        Vector gradientMean = calculateColwiseMean(gradients);
//        std::cout << "gradientMean" << gradientMean << std::endl;
        nn.backprop(optimizer, gradientMean);

        std::cout << "Epoch " << epoch << ", Loss: " << calculateMean(losses)
                  << std::endl;
    }
    return 1;
}

int main()
{
    CAnyLayer layer1 = LinearLayer(784, 300);
    CAnyLayer activation1 = SigmoidLayer(300);
    CAnyLayer layer2 = LinearLayer(300, 100);
    CAnyLayer activation2 = SigmoidLayer(100);
    CAnyLayer layer3 = LinearLayer(100, 10);
    CAnyLayer activation3 = SoftmaxLayer();

    NeuralNetwork nn;
    nn.addLayer(layer1);
    nn.addLayer(activation1);
    nn.addLayer(layer2);
    nn.addLayer(activation2);
    nn.addLayer(layer3);
    nn.addLayer(activation3);

    MomentumOptimizer optimizer(0.01, 0.9);  // Learning rate and momentum

    MSELoss loss;


    DataHandler trainHandler;
    trainHandler.readData(
            "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_test.csv");
    std::cout << "rows read by dataHandler: " << trainHandler.getNumberOfSamples()
              << std::endl;


    int batchSize = 5;
    int epochs = 10000;

    trainNN(nn, loss, optimizer, trainHandler, batchSize, epochs);


    return 0;
}





