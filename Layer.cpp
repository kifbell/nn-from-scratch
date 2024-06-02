//
// Created by Kirill Belyakov on 4/18/24.
//

#include "Layer.h"
#include "Optimizer.h"
#include <iostream>
#include <cassert>

namespace NeuralNet
{

Matrix LinearLayer::passForward(const Matrix &input)
{
    assert((input.rows() == deltaA.cols()) && "Gradient shape misalignment.");
    zCache_ = input;
    int batchSize = input.cols();
    Matrix bias = bias_.replicate(1, batchSize);
    Matrix ans = weights_ * input + bias;
    return weights_ * input + bias;
}

Vector LinearLayer::backprop(Optimizer &optimizer, const Vector &u)
{
    Matrix deltaA = Matrix::Zero(weights_.rows(), weights_.cols());
    int batchSize = zCache_.cols();

    for (size_t idx = 0;
         idx < batchSize; ++idx)  // try suboptimal implementation for now
    {
        Matrix grad = u * zCache_.col(idx).transpose();
        assert(
                ((grad.rows() == deltaA.rows()) && grad.cols() == deltaA.cols()) &&
                "Gradient shape misalignment.");
        deltaA = deltaA + grad;
    }
    deltaA = deltaA / batchSize;
    Vector deltaB = u;  // (dσ)^T * u^T

    Vector u_bar = weights_.transpose() * u;  // u * dσ * A
    optimizer.update(
            weights_,
            bias_,
            deltaA,
            deltaB,
            optimizerState_
    );

    return u_bar;
}

Vector softMaxVector(const Matrix &input)
{
    double maxInput = input.maxCoeff();

    Vector exps = (input.array() - maxInput).exp();

    return exps / exps.sum();
}

Matrix SoftmaxLayer::passForward(const Matrix &input)
{
    zCache_ = input;
    Matrix ans = Matrix::Zero(input.rows(), input.cols());

    // Apply the softmax function to each column
    for (int i = 0; i < input.cols(); ++i)
    {
        ans.col(i) = softMaxVector(input.col(i));
    }
    return ans;
}

Vector SoftmaxLayer::backprop(Optimizer &, const Vector &u)
{
    Matrix jacobian = Matrix(u.rows(), u.rows());
    int batchSize = zCache_.cols();
    Matrix output = passForward(zCache_);
    for (size_t idx = 0; idx < batchSize; ++idx)
    {
        Matrix diagonal = output.col(idx).asDiagonal();
        jacobian = jacobian + diagonal - output.col(idx) * output.col(idx).transpose();
    }
    return jacobian * u / batchSize; // fixme isn't it u * sigma
}
}

