//
// Created by Kirill Belyakov on 4/22/24.
//

#ifndef NN_FROM_SCRATCH_LOSS_H
#define NN_FROM_SCRATCH_LOSS_H

#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>

namespace NeuralNet
{
using Vector = Eigen::VectorXd;

class MSELoss
{
public:
    double
    compute_loss(const Vector &predictions, const Vector &targets);

    Vector
    compute_gradient(const Vector &predictions, const Vector &targets);
};


}
#endif //NN_FROM_SCRATCH_LOSS_H
