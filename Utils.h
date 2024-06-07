//
// Created by Kirill Belyakov on 5/25/24.
//

#ifndef NN_FROM_SCRATCH_UTILS_H
#define NN_FROM_SCRATCH_UTILS_H

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

namespace NeuralNet
{
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

int argmax(const Vector &vec);

Vector labelToOneHot(int label, const int numClasses);

Matrix labelsToOneHot(const Vector &labels, const int numClasses);

Vector calculateColwiseMean(const Eigen::MatrixXd &matrix);

int countCorrectPredictions(const Matrix &predictions, const Matrix &targetVectors);

Vector calculateRowwiseMean(const Eigen::MatrixXd &matrix);
}// namespace NeuralNet

#endif//NN_FROM_SCRATCH_UTILS_H
