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
//    std::cout << "MomentumOptimizer update" << std::endl;

    if (optimizerState.find("velocityWeights") == optimizerState.end())
    {
        optimizerState["velocityWeights"] = Eigen::MatrixXd::Zero(weights.rows(),
                                                                  weights.cols());
    }
    if (optimizerState.find("velocityBiases") == optimizerState.end())
    {
        optimizerState["velocityBiases"] = Eigen::VectorXd::Zero(biases.size());
    }

    optimizerState["velocityWeights"] =
            momentum * optimizerState["velocityWeights"] +
            (1 - momentum) * gradientWeights;
    optimizerState["velocityBiases"] =
            momentum * optimizerState["velocityBiases"] +
            (1 - momentum) * gradientBiases;

//    std::cout << "velocityWeights" << std::endl;
//    std::cout << optimizerState["velocityWeights"] << std::endl;

    weights -= learningRate * optimizerState["velocityWeights"];
//    std::cout <<  optimizerState["velocityWeights"] <<std::endl;
//    std::cout <<  gradientWeights <<std::endl;
    biases -= learningRate * optimizerState["velocityBiases"];
}