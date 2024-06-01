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
//    std::cout << "forward zCache_.transpose()" << std::endl;

//    std::cout << zCache_.transpose() << std::endl;
//    std::cout << "weights" << std::endl;
//    std::cout << weights << std::endl;
//    assert(0);
    zCache_ = input;
//    std::cout << "LinearInput.shape " << input.rows() << ' ' << input.cols() << std::endl;
//    std::cout << "LinearInput.shape " << input.transpose() << std::endl;
    int batchSize = input.cols();
    Matrix bias = bias_.replicate(1, batchSize);
//    M*N N*K + M
    Matrix ans = weights_ * input + bias;
//    std::cout << "Linear and  " << ans.transpose() << std::endl;
    return weights_ * input + bias;
}

Vector LinearLayer::backprop(Optimizer &optimizer, const Vector &u)
{
//    std::cout << "LinearLayer.backprop" << std::endl;
    // Store original weights and biases for later comparison
//    Matrix originalWeights = weights_;
//    Vector originalBiases = bias_;

//    Matrix deltaA = u * zCache_.transpose();  // (dσ)^T * u^T * z^T
    Matrix deltaA = Matrix::Zero(weights_.rows(), weights_.cols());
    int batchSize = zCache_.cols();
//    std::cout << "zCache_.transpose() " << zCache_.transpose() << std::endl;
//    std::cout <<"u  " <<  u << std::endl;
//    std::cout <<"u shape " <<  u.rows()<< " " << u.cols()<< std::endl;

    for (size_t idx = 0;
         idx < batchSize; ++idx)  // try suboptimal implementation for now
    {
        Matrix grad = u * zCache_.col(idx).transpose();
//        std::cout <<"grad shape " <<  grad.rows()<< " " << grad.cols()<< std::endl;
        assert(
                ((grad.rows() == deltaA.rows()) && grad.cols() == deltaA.cols()()) &&
                "Weights should have changed after optimization.");
        deltaA = deltaA + grad;
    }

//    std::cout << "u.shape: " << u.rows() << " " <<u.cols() << std::endl;
//    std::cout << "zCache_.col(0).shape: " << zCache_.col(0).rows() << " " << zCache_.col(0).cols() << std::endl;
//    std::cout << "(u* zCache_.col(0).transpose()).shape: " << (u* zCache_.col(0).transpose()).rows() << " " << (u* zCache_.col(0).transpose()).cols() << std::endl;

    deltaA = deltaA / batchSize;
    Vector deltaB = u;  // (dσ)^T * u^T
//    std::cout << "deltaA: " << deltaA << std::endl;
//    std::cout << "deltaB.transpose(): " << deltaB.transpose() << std::endl;
//    std::cout << u.transpose() << std::endl;
//    std::cout << "u.transpose()" << std::endl;
//    std::cout << u.transpose() << std::endl;
//    std::cout << "zCache_.transpose()" << std::endl;
//    std::cout << zCache_.transpose() << std::endl;

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
//    std::cout << "softmaxInput.shape " << input.rows() << ' ' << input.cols() << std::endl;
//    std::cout << "softmax.shape " << ans.rows() << ' ' << ans.cols() << std::endl;

    return ans;
}


//Matrix SoftmaxLayer::getJacobian(const Matrix &output)
//{
////    std::cout <<"getJacobian output "<< std::endl;
////    std::cout << output.cols()<< std::endl;
//
//    assert(output.cols() == 1 && "only n by 1 matrices are accepted.");
//
//    int dim = output.size();
//    Matrix jacobian(dim, dim);
//
//    for (int i = 0; i < dim; ++i)
//    {
//        for (int j = 0; j < dim; ++j)
//        {
//            if (i == j)
//            {
//                jacobian(i, j) = output(i) * (1 - output(i));
//            } else
//            {
//                jacobian(i, j) = -output(i) * output(j);
//            }
//        }
//    }
//    return jacobian;
//}


Vector SoftmaxLayer::backprop(Optimizer &, const Vector &u)
{
    Matrix jacobian = Matrix(u.rows(), u.rows());
    int batchSize = zCache_.cols();
    Matrix output = passForward(zCache_);
    for (size_t idx = 0; idx < batchSize; ++idx)
    {
//        jacobian = jacobian + getJacobian(output.col(idx));
        Matrix diagonal = output.col(idx).asDiagonal();
        jacobian = jacobian + diagonal - output.col(idx) * output.col(idx).transpose();
    }
    return jacobian * u / batchSize; // fixme isn't it u * sigma
}
}

