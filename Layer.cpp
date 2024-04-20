//
// Created by Kirill Belyakov on 4/18/24.
//

#include "Layer.h"
#include "Optimizer.h"
#include <iostream>

Eigen::VectorXd LinearLayer::passForward(const Eigen::VectorXd &input)
{
    z_cache = input;  // Cache the input (z) for use in backpropagation
//    std::cout << "forward z_cache.transpose()" << std::endl;
//    std::cout << z_cache.transpose() << std::endl;
//    std::cout << "weights" << std::endl;
//    std::cout << weights << std::endl;
//    assert(0);
    return weights * input + bias;  // Wx + b
};

Eigen::VectorXd LinearLayer::backprop(Optimizer &optimizer, const Eigen::VectorXd &u)
{
//    std::cout << "LinearLayer.backprop" << std::endl;
    // Store original weights and biases for later comparison
    Eigen::MatrixXd originalWeights = this->weights;
    Eigen::VectorXd originalBiases = this->bias;

    Eigen::MatrixXd deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
    Eigen::VectorXd deltaB = u;  // (dσ)^T * u^T
//    std::cout << "u.transpose()" << std::endl;
//    std::cout << u.transpose() << std::endl;
//    std::cout << "z_cache.transpose()" << std::endl;
//    std::cout << z_cache.transpose() << std::endl;

    optimizer.update(
            this->weights,
            this->bias,
            deltaA,
            deltaB,
            this->optimizerState
    );

    assert(!this->weights.isApprox(originalWeights) &&
           "Weights should have changed after optimization.");
    assert(!this->bias.isApprox(originalBiases) &&
           "Biases should have changed after optimization.");

    Eigen::VectorXd u_bar = weights.transpose() * u;  // u * dσ * A

    return u_bar;
};


Eigen::VectorXd ReLULayer::passForward(const Eigen::VectorXd &input)
{
    // Calculate the output using ReLU activation
    Eigen::MatrixXd output = input.unaryExpr(
            [](double elem) { return std::max(0.0, elem); });

    // Recalculate dsigma based on the input
    dsigma = input.unaryExpr(
            [](double elem) { return elem > 0 ? 1.0 : 0.0; }).asDiagonal();

    return output;
}

Eigen::VectorXd ReLULayer::backprop(Optimizer &optimizer, const Eigen::VectorXd &u)
{
    optimizer;
    // Apply the stored dsigma to the incoming gradient
    return dsigma * u;
}


Eigen::VectorXd SigmoidLayer::passForward(const Eigen::VectorXd &input)
{
    Eigen::VectorXd output = input.unaryExpr([](double elem) {
        return 1.0 / (1.0 + std::exp(-elem));
    }); // fixme this is 1, dsigma is then 0

//    std::cout << "output.transpose()" << std::endl;
//    std::cout << output.transpose() << std::endl;

    // Recalculate dsigma based on the output of the sigmoid function
    dsigma = output.unaryExpr(
            [](double y) {
        return y * (1.0 - y);
    }).asDiagonal();

    return output;
}

Eigen::VectorXd SigmoidLayer::backprop(Optimizer &optimizer, const Eigen::VectorXd &u)
{
    optimizer;

//    std::cout << "dsigma" << std::endl;
//    std::cout << dsigma << std::endl;

    return dsigma * u;
}

