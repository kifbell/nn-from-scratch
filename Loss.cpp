//
// Created by Kirill Belyakov on 4/22/24.
//

#include <cmath>
#include "Loss.h"

namespace NeuralNet
{

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

Vector
MSELoss::computeLoss(
        const Matrix &predictions,
        const Matrix &targets
)
{
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols())
    {
        throw std::invalid_argument(
                "Predictions and targets matrices must have the same dimensions.");
    }

    // Calculate the element-wise difference
    Matrix diff = predictions - targets;

    // Calculate the sum of squares for each column
    Vector sumOfSquares = diff.array().square().colwise().sum();

    // Calculate the mean of the sum of squares
    return sumOfSquares;
}


Matrix MSELoss::computeGradient(
        const Matrix &predictions,
        const Matrix &targets
)
{

    return 2.0 * (predictions - targets);

}
}