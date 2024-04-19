#ifndef LEETCODE_OPTIMIZER_H
#define LEETCODE_OPTIMIZER_H

#include <Eigen/Dense>
#include "OptimizerState.h"
#include <memory>
//#include "Layer.h"


class Optimizer
{
private:

    OptimizerState state;
public:
    Optimizer() {};

    virtual void update(
//            Layer& layer
            Eigen::MatrixXd &weights,
            Eigen::VectorXd &bias,
            const Eigen::MatrixXd &gradientWeights,
            const Eigen::VectorXd &gradientBiases,
            std::unique_ptr<OptimizerState>& optimizerState
    ) {
        std::cout << "Optimizer update" << std::endl;
    };

    ~Optimizer()
    {}
};




// Forward declaration of the optimizer state
struct MomentumOptimizerState : public OptimizerState
{
    Eigen::MatrixXd velocityWeights;
    Eigen::VectorXd velocityBiases;

    MomentumOptimizerState(int weights_rows, int weights_cols, int biases_size)
            : velocityWeights(Eigen::MatrixXd::Zero(weights_rows, weights_cols)),
              velocityBiases(Eigen::VectorXd::Zero(biases_size))
    {}
};


class MomentumOptimizer : public Optimizer {
private:
    double learningRate;
    double momentum;
    std::unique_ptr<MomentumOptimizerState> state;  // Encapsulates the state

public:

    MomentumOptimizer(double lr, double m) : learningRate(lr), momentum(m)
    {};

    void update(Eigen::MatrixXd& weights,
                Eigen::VectorXd& biases,
                const Eigen::MatrixXd& gradientWeights,
                const Eigen::VectorXd& gradientBiases,
                std::unique_ptr<MomentumOptimizerState>& optimizerState
    );

    // Additional methods as necessary
};

#endif //LEETCODE_OPTIMIZER_H
