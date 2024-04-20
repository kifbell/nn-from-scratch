#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "Layer.h"
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <iostream>

class SoftmaxLayer : public Layer
{
private:
    Eigen::VectorXd lastInput;
    Eigen::VectorXd lastOutput;

public:
    SoftmaxLayer()
    {}

    ~SoftmaxLayer()
    {}

    Eigen::VectorXd passForward(const Eigen::VectorXd &input) override;

    Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u) override;
};


Eigen::VectorXd SoftmaxLayer::passForward(const Eigen::VectorXd &input)
{
//    lastInput = input; // Store the input for use in backprop
//    Eigen::VectorXd exps = input.unaryExpr([](double elem) { return std::exp(elem); });
//    double sum = exps.sum();
//    lastOutput = exps / sum; // Compute softmax probabilities
//    return lastOutput;
    double maxInput = input.maxCoeff();
    Eigen::VectorXd exps = (input.array() - maxInput).exp();

    double sum = exps.sum();
    lastOutput = exps / sum;  // Compute softmax probabilities
    return lastOutput;
}

Eigen::VectorXd SoftmaxLayer::backprop(Optimizer &optimizer, const Eigen::VectorXd &u)
{
    optimizer; // Suppress unused variable warning if not used

    // Compute the Jacobian matrix of the softmax function
    int dim = lastOutput.size();
    Eigen::MatrixXd jacobian(dim, dim);

    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            if (i == j)
            {
                jacobian(i, j) = lastOutput(i) * (1 - lastOutput(i));
            } else
            {
                jacobian(i, j) = -lastOutput(i) * lastOutput(j);
            }
        }
    }

    // Compute the gradient for the layer inputs
    Eigen::VectorXd gradient = jacobian * u;
//    std::cout << "gradient" << std::endl;
//    std::cout << gradient << std::endl;

    return gradient;
}


#endif // SOFTMAX_LAYER_H
