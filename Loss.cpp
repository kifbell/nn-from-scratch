//
// Created by Kirill Belyakov on 4/22/24.
//

#include "Loss.h"

namespace NeuralNet
{

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

double
MSELoss::compute_loss(
        const Vector &predictions,
        const Vector &targets
)
{
    return (predictions - targets).array().square().mean();
}


Vector MSELoss::compute_gradient(
        const Vector &predictions,
        const Vector &targets
)
{

    return 2.0 * (predictions - targets) / predictions.cols();

}
}