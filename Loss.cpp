//
// Created by Kirill Belyakov on 4/22/24.
//

#include "Loss.h"
#include <cmath>

namespace NeuralNet
{

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

Vector
MSELoss::computeLoss(
        const Matrix &predictions,
        const Matrix &targets)
{
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols())
    {
        throw std::invalid_argument(
                "Predictions and targets matrices must have the same dimensions.");
    }

    Matrix diff = predictions - targets;
    Vector sumOfSquares = diff.array().square().colwise().sum();

    return sumOfSquares;
}


Matrix MSELoss::computeGradient(
        const Matrix &predictions,
        const Matrix &targets)
{

    return 2.0 * (predictions - targets);
}
}// namespace NeuralNet