#include "Optimizer.h"
#include <Eigen/Dense>





    // todo initially I wanted to pass Layer & layer,  but i bumped into cycling include
//    void update(Layer &layer)
//    {
//        if (layer.optimizerState == nullptr)
//        {
//
//            layer.optimizerState = std::make_unique<MomentumOptimizerState>(
//                    MomentumOptimizerState(layer.getWeights().rows(),
//                                           layer.getWeights().cols(),
//                                           layer.getBiases().size()));
//
//        }
//        layer.optimizerState->velocityWeights =
//                momentum * layer.optimizerState->velocityWeights + (1 - momentum) * gradientWeights;
//        layer.optimizerState->velocityBiases =
//                momentum * layer.optimizerState->velocityBiases + (1 - momentum) * gradientBiases;
//
//        layer.getWeights() -= learningRate * layer.optimizerState->velocityWeights;
//        layer.getBiases() -= learningRate * layer.optimizerState->velocityBiases;
//        }

void MomentumOptimizer::update(
            Eigen::MatrixXd &weights,
                Eigen::VectorXd &biases,
                const Eigen::MatrixXd &gradientWeights,
                const Eigen::VectorXd &gradientBiases,
//                const Eigen::VectorXd &u,
                std::unique_ptr<MomentumOptimizerState>& optimizerState
                )
    {
        if (optimizerState == nullptr)
        {
//            velocityWeights:Eigen::MatrixXd::Zero(weights.rows(),
//                                                  weights.cols()),
//                velocityBiases:  Eigen::VectorXd::Zero(biases.size()),
            optimizerState = std::make_unique<MomentumOptimizerState>(
//            std::unique_ptr<MomentumOptimizerState> mewOptimizerState = std::make_unique<MomentumOptimizerState>(
                    MomentumOptimizerState(weights.rows(), weights.cols(), biases.size()));
//            optimizerState = mewOptimizerState;

        }

        // Update velocities
        optimizerState->velocityWeights =
                momentum * optimizerState->velocityWeights + (1 - momentum) * gradientWeights;
        optimizerState->velocityBiases =
                momentum * optimizerState->velocityBiases + (1 - momentum) * gradientBiases;

        weights -= learningRate * optimizerState->velocityWeights;
        biases -= learningRate * optimizerState->velocityBiases;
    }
