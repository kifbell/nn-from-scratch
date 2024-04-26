#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <Eigen/Dense>
#include <memory>
#include <map>
#include "eigen/Eigen/Core"

//#include "Layer.h"


class Optimizer
{
private:
    double lr_;
public:
    virtual void update(
            Eigen::MatrixXd &weights,
            Eigen::VectorXd &bias,
            const Eigen::MatrixXd &gradientWeights,
            const Eigen::VectorXd &gradientBiases,
            std::map<std::string, Eigen::MatrixXd> &optimizerState
    ) = 0;

    void lrUpdate(double lrNew)
    { lr_ = lrNew; }

};

class MomentumOptimizer : public Optimizer
{
private:
    double lr_;
    double momentum;
public:

    MomentumOptimizer(double lr, double m) : lr_(lr), momentum(m)
    {

    }

    ~MomentumOptimizer() = default;

    void update(Eigen::MatrixXd &weights,
                Eigen::VectorXd &biases,
                const Eigen::MatrixXd &gradientWeights,
                const Eigen::VectorXd &gradientBiases,
                std::map<std::string, Eigen::MatrixXd> &optimizerState
    ) override;

};

#endif //NN_OPTIMIZER_H
