#include "Optimizer.h"
#include "iostream"
#include <Eigen/Dense>


void MomentumOptimizer::update(Eigen::MatrixXd &weights,
                               Eigen::VectorXd &biases,
                               const Eigen::MatrixXd &gradientWeights,
                               const Eigen::VectorXd &gradientBiases,
                               OptimizerState &optimizerState
)
{
    Eigen::MatrixXd& velocityWeights = optimizerState.getMatrix("velocityWeights", Eigen::MatrixXd::Zero(weights.rows(), weights.cols()));
    Eigen::VectorXd& velocityBiases = optimizerState.getVector("velocityBiases", Eigen::VectorXd::Zero(biases.size()));

    velocityWeights = momentum * velocityWeights + (1 - momentum) * gradientWeights;
    velocityBiases = momentum * velocityBiases + (1 - momentum) * gradientBiases;

    weights -= lr_ * velocityWeights;
    biases -= lr_ * velocityBiases;
}

void AMSGrad::update(Eigen::MatrixXd &weights,
                     Eigen::VectorXd &biases,
                     const Eigen::MatrixXd &gradientWeights,
                     const Eigen::VectorXd &gradientBiases,
                     OptimizerState &optimizerState)
{
    t++;

    Eigen::MatrixXd& mWeights = optimizerState.getMatrix("mWeights", Eigen::MatrixXd::Zero(weights.rows(), weights.cols()));
    Eigen::MatrixXd& vWeights = optimizerState.getMatrix("vWeights", Eigen::MatrixXd::Zero(weights.rows(), weights.cols()));
    Eigen::MatrixXd& vHatWeights = optimizerState.getMatrix("vHatWeights", Eigen::MatrixXd::Zero(weights.rows(), weights.cols()));

    Eigen::VectorXd& mBiases = optimizerState.getVector("mBiases", Eigen::VectorXd::Zero(biases.size()));
    Eigen::VectorXd& vBiases = optimizerState.getVector("vBiases", Eigen::VectorXd::Zero(biases.size()));
    Eigen::VectorXd& vHatBiases = optimizerState.getVector("vHatBiases", Eigen::VectorXd::Zero(biases.size()));

    mWeights = beta1 * mWeights + (1 - beta1) * gradientWeights;
    vWeights = beta2 * vWeights + (1 - beta2) * gradientWeights.array().square().matrix();
    vHatWeights = vWeights.cwiseMax(vHatWeights);

    mBiases = beta1 * mBiases + (1 - beta1) * gradientBiases;
    vBiases = beta2 * vBiases + (1 - beta2) * gradientBiases.array().square().matrix();
    vHatBiases = vBiases.cwiseMax(vHatBiases);

    Eigen::MatrixXd mWeightsCorrected = mWeights / (1 - std::pow(beta1, t));
    Eigen::MatrixXd vHatWeightsCorrected = vHatWeights / (1 - std::pow(beta2, t));
    Eigen::VectorXd mBiasesCorrected = mBiases / (1 - std::pow(beta1, t));
    Eigen::VectorXd vHatBiasesCorrected = vHatBiases / (1 - std::pow(beta2, t));

    weights -= lr_ * (mWeightsCorrected.array() / (vHatWeightsCorrected.array().sqrt() + epsilon)).matrix();
    biases -= lr_ * (mBiasesCorrected.array() / (vHatBiasesCorrected.array().sqrt() + epsilon)).matrix();
}
