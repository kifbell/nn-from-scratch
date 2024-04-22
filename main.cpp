#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>

#include "Layer.h"
#include "Optimizer.h"
#include "DataHandler.h"
#include "SoftmaxLayer.h"


class MSELoss
{
public:
    double
    compute_loss(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets)
    {
        return (predictions - targets).array().square().mean();
    }

    Eigen::MatrixXd
    compute_gradient(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets)
    {
        return 2.0 * (predictions - targets) / predictions.cols();
    }
};


class NeuralNetwork
{
private:
    std::vector<std::shared_ptr<Layer>> layers;

public:
    NeuralNetwork(const std::vector<std::shared_ptr<Layer>> &init_layers) : layers(
            init_layers)
    {}

    Eigen::VectorXd passForward(const Eigen::VectorXd &input)
    {
        Eigen::VectorXd current_output = input;
        for (auto &layer: layers)
        {
            current_output = layer->passForward(current_output);
        }
        return current_output;
    }

    void backprop(Optimizer &optimizer, const Eigen::VectorXd &output_gradient)
    {
        Eigen::VectorXd current_gradient = output_gradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            current_gradient = (*it)->backprop(optimizer, current_gradient);
        }
    }
};


int argmax(const Eigen::VectorXd &vec)
{
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return maxIndex;
}


Eigen::VectorXd labelToOneHot(int label)
{
    Eigen::VectorXd oneHot = Eigen::VectorXd::Zero(
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

Eigen::VectorXd calculateMean(const std::vector<Eigen::VectorXd> &vectors)
{
    if (vectors.empty()) return Eigen::VectorXd();

    Eigen::VectorXd sum = Eigen::VectorXd::Zero(
            vectors[0].size());  // Initialize sum vector of the same size
    for (const auto &vec: vectors)
    {
        sum += vec;
    }
    return sum / vectors.size();  // Divide by the number of vectors to get the mean
}

int main()
{
    std::shared_ptr<Layer> layer1 = std::make_shared<LinearLayer>(784, 300);
    std::shared_ptr<Layer> activation1 = std::make_shared<SigmoidLayer>(300);
    std::shared_ptr<Layer> layer2 = std::make_shared<LinearLayer>(300, 100);
    std::shared_ptr<Layer> activation2 = std::make_shared<SigmoidLayer>(100);
    std::shared_ptr<Layer> layer3 = std::make_shared<LinearLayer>(100, 10);
    std::shared_ptr<Layer> activation3 = std::make_shared<SoftmaxLayer>();

    std::vector<std::shared_ptr<Layer>> layers = {
            layer1,
            activation1,
            layer2,
            activation2,
            layer3,
            activation3
    };
    NeuralNetwork nn(layers);

    MomentumOptimizer optimizer(0.01, 0.9);  // Learning rate and momentum

    MSELoss loss;


    NeuralNet::DataHandler trainHandler;
    trainHandler.readData(
            "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_train.csv");
    std::cout << "rows read by dataHandler: " << trainHandler.getNumberOfSamples()
              << std::endl;

    int batchSize = 20;
    int epochs = 10000;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        auto batch = trainHandler.getRandomBatch(batchSize);


        std::vector<Eigen::VectorXd> gradients;
        std::vector<double> losses;
        for (size_t idx = 0; idx < batchSize; idx++)
        {
            Eigen::VectorXd input(batch.second.row(idx).size());
            Eigen::VectorXd target = labelToOneHot(batch.first[idx]);

            Eigen::VectorXd prediction = nn.passForward(input);

//            std::cout << "target: " << batch.first[idx] << ", prediction: " << argmax(prediction)<< std::endl;
//            std::cout << "prediction: " << prediction << std::endl;
//            std::cout << "    target: " << target << std::endl;

            double current_loss = loss.compute_loss(prediction, target);
//            std::cout << "current_loss: " <<current_loss << std::endl;
            Eigen::VectorXd gradient = loss.compute_gradient(prediction, target);

            // batch loss history
            gradients.push_back(gradient);
            losses.push_back(current_loss);
        }

        // step by mean batch gradient
        Eigen::VectorXd gradientMean = calculateMean(gradients);
        nn.backprop(optimizer, gradientMean);

        std::cout << "Epoch " << epoch << ", Loss: " << calculateMean(losses)
                  << std::endl;
    }

    return 0;
}





