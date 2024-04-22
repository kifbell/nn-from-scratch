//
// Created by Kirill Belyakov on 4/18/24.
//

#include "Layer.h"
#include "Optimizer.h"
#include <iostream>

namespace NeuralNet
{
Vector LinearLayer::passForward(const Vector &input)
{
    z_cache = input;  // Cache the input (z) for use in backpropagation
//    std::cout << "forward z_cache.transpose()" << std::endl;
//    std::cout << z_cache.transpose() << std::endl;
//    std::cout << "weights" << std::endl;
//    std::cout << weights << std::endl;
//    assert(0);
    return weights_ * input + bias_;
};

Vector LinearLayer::backprop(Optimizer &optimizer, const Vector &u)
{
//    std::cout << "LinearLayer.backprop" << std::endl;
    // Store original weights and biases for later comparison
    Matrix originalWeights = weights_;
    Vector originalBiases = bias_;

    Matrix deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
    Vector deltaB = u;  // (dσ)^T * u^T
//    std::cout << deltaA << std::endl;
//    std::cout << u.transpose() << std::endl;
//    std::cout << "u.transpose()" << std::endl;
//    std::cout << u.transpose() << std::endl;
//    std::cout << "z_cache.transpose()" << std::endl;
//    std::cout << z_cache.transpose() << std::endl;

    Vector u_bar = weights_.transpose() * u;  // u * dσ * A
    optimizer.update(
            weights_,
            bias_,
            deltaA,
            deltaB,
            optimizerState_
    );

    assert(!weights_.isApprox(originalWeights) &&
           "Weights should have changed after optimization.");
    assert(!bias_.isApprox(originalBiases) &&
           "Biases should have changed after optimization.");


    return u_bar;
};


Vector ReLULayer::passForward(const Vector &input)
{
    Matrix output = input.unaryExpr(
            [](double elem) { return std::max(0.0, elem); });

    dsigma = input.unaryExpr(
            [](double elem) { return elem > 0 ? 1.0 : 0.0; }).asDiagonal();

    return output;
}

Vector ReLULayer::backprop(Optimizer &, const Vector &u)
{
    return dsigma * u;
}


Vector SigmoidLayer::passForward(const Vector &input)
{
    Vector output = input.unaryExpr([](double elem) {
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

Vector SigmoidLayer::backprop(Optimizer &, const Vector &u)
{
//    std::cout << "dsigma" << std::endl;
//    std::cout << dsigma << std::endl;
    return dsigma * u;
}


Vector SoftmaxLayer::passForward(const Vector &input)
{
    lastInput_ = input;
    double maxInput = input.maxCoeff();
    Vector exps = (input.array() - maxInput).exp();

    return exps / exps.sum();
}



Matrix SoftmaxLayer::getJacobian(Vector& output){

    int dim = output.size();
    Matrix jacobian(dim, dim);

    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            if (i == j)
            {
                jacobian(i, j) = output(i) * (1 - output(i));
            } else
            {
                jacobian(i, j) = -output(i) * output(j);
            }
        }
    }
    return jacobian;
}


Vector SoftmaxLayer::backprop(Optimizer &, const Vector &u)
{
    Vector lastOutput = passForward(lastInput_);
    Matrix jacobian = getJacobian(lastOutput);
    return jacobian * u;
}


}

