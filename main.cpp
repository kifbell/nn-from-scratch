#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>


//#include <eigen/Eigen/Dense>
#include "Layer.h"
#include "Optimizer.h"


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
    // Constructor that accepts a vector of layers
    NeuralNetwork(const std::vector<std::shared_ptr<Layer>> &init_layers) : layers(
            init_layers)
    {}

    // Forward pass through all layers
    Eigen::VectorXd pass_forward(const Eigen::VectorXd &input)
    {
        Eigen::VectorXd current_output = input;
        for (auto &layer: layers)
        {
            current_output = layer->pass_forward(current_output);
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




int main() {
    // Create layers
    std::shared_ptr<Layer> layer1 = std::make_shared<LinearLayer>(3, 5); // 3 inputs, 5 outputs
    std::shared_ptr<Layer> layer2 = std::make_shared<LinearLayer>(5, 1); // 5 inputs, 1 output

    // Create neural network
    std::vector<std::shared_ptr<Layer>> layers = {layer1, layer2};
    NeuralNetwork nn(layers);

    // Optimizer
    MomentumOptimizer optimizer(0.1, 0.9);  // Learning rate and momentum

    // Loss
    MSELoss loss;

    // Example training data
    Eigen::VectorXd input(3);  // Input size must match the first layer's input size
    input << 0.5, -1.5, 0.6;
    Eigen::VectorXd target(1); // Target output size must match the last layer's output size
    target << 0.1;

    // Training loop
    for (int i = 0; i < 10; ++i) { // Train for 100 iterations
        // Forward pass
        Eigen::VectorXd prediction = nn.pass_forward(input);
        std::cout << prediction<< std::endl;

        // Compute loss and gradient
        double current_loss = loss.compute_loss(prediction, target);
        Eigen::VectorXd loss_gradient = loss.compute_gradient(prediction, target);

        // Backpropagation
        nn.backprop(optimizer, loss_gradient);

        // Output training progress
        std::cout << "Epoch " << i + 1 << ", Loss: " << current_loss << std::endl;
    }

    return 0;
}


//int main()
//{
//    auto layer = LinearLayer(10, 5);
//    MomentumOptimizer opt = MomentumOptimizer(0.5, 0.9);
//
//    Eigen::VectorXd z = Eigen::VectorXd::Random(10, 1);
//    Eigen::VectorXd y_hat = layer.pass_forward(z);
//
////    std::cout << y_hat << '\n';
//
//
//    Eigen::VectorXd u = Eigen::VectorXd::Random(5, 1); // Example gradient matrix
//    Eigen::VectorXd u_bar = layer.backprop(opt, u);
//
//
//    std::cout << u_bar << '\n';
//
//
//    return 0;
//}
