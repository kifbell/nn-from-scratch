#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <Eigen/Dense>
#include "OptimizerState.h"
#include <memory>
#include <map>
#include "eigen/Eigen/Core"

//#include "Layer.h"


class Optimizer
{
public:
    Optimizer() =default;
     ~Optimizer()= default;

    virtual void update(
//            Layer& layer
            Eigen::MatrixXd &weights,
            Eigen::VectorXd &bias,
            const Eigen::MatrixXd &gradientWeights,
            const Eigen::VectorXd &gradientBiases,
            std::map<std::string, Eigen::MatrixXd> &optimizerState
    ) = 0;

};

class MomentumOptimizer : public Optimizer
{
private:
    double learningRate;
    double momentum;
//    std::unique_ptr<MomentumOptimizerState> state;  // Encapsulates the state

public:

    MomentumOptimizer(double lr, double m) : learningRate(lr), momentum(m)
    {

    }

    ~MomentumOptimizer() = default;

    void update(Eigen::MatrixXd &weights,
                Eigen::VectorXd &biases,
                const Eigen::MatrixXd &gradientWeights,
                const Eigen::VectorXd &gradientBiases,
                std::map<std::string, Eigen::MatrixXd> &optimizerState
    ) override;

    // Additional methods as necessary
};

#endif //NN_OPTIMIZER_H
