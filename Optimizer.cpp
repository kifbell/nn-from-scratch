#include "Optimizer.h"
#include "iostream"
#include <Eigen/Dense>


void MomentumOptimizer::update(Eigen::MatrixXd &weights,
                               Eigen::VectorXd &biases,
                               const Eigen::MatrixXd &gradientWeights,
                               const Eigen::VectorXd &gradientBiases,
                               std::map<std::string, Eigen::MatrixXd> &optimizerState
)
{
    std::cout << "MomentumOptimizer update" << std::endl;

    if (optimizerState.find("velocityWeights") == optimizerState.end())
    {
        optimizerState["velocityWeights"] = Eigen::MatrixXd::Zero(weights.rows(),
                                                                  weights.cols());
    }
    if (optimizerState.find("velocityBiases") == optimizerState.end())
    {
        optimizerState["velocityBiases"] = Eigen::VectorXd::Zero(biases.size());
    }
    // Update velocities
    optimizerState["velocityWeights"] =
            momentum * optimizerState["velocityWeights"] +
            (1 - momentum) * gradientWeights;
    optimizerState["velocityBiases"] =
            momentum * optimizerState["velocityBiases"] +
            (1 - momentum) * gradientBiases;

    weights -= learningRate * optimizerState["velocityWeights"];
    biases -= learningRate * optimizerState["velocityBiases"];
}