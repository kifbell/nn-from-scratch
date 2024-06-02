#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <Eigen/Dense>
#include <memory>
#include <map>
#include "eigen/Eigen/Core"
#include <string>

class OptimizerState
{
public:
    Eigen::MatrixXd &getMatrix(const std::string &key,
                               const Eigen::MatrixXd &defaultMatrix = Eigen::MatrixXd())
    {
        if (matrixState.find(key) == matrixState.end())
        {
            matrixState[key] = defaultMatrix;
        }
        return matrixState[key];
    }

    Eigen::VectorXd &getVector(const std::string &key,
                               const Eigen::VectorXd &defaultVector = Eigen::VectorXd())
    {
        if (vectorState.find(key) == vectorState.end())
        {
            vectorState[key] = defaultVector;
        }
        return vectorState[key];
    }

private:
    std::map<std::string, Eigen::MatrixXd> matrixState;
    std::map<std::string, Eigen::VectorXd> vectorState;
};

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
            OptimizerState &optimizerState
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
                OptimizerState &optimizerState
    ) override;

};


class AMSGrad : public Optimizer
{
private:
    double lr_;
    double beta1;
    double beta2;
    double epsilon;
    int t; // timestep
public:
    AMSGrad(double learningRate, double beta1, double beta2, double epsilon)
            : lr_(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0)
    {}

    ~AMSGrad() = default;

    void update(Eigen::MatrixXd &weights,
                Eigen::VectorXd &biases,
                const Eigen::MatrixXd &gradientWeights,
                const Eigen::VectorXd &gradientBiases,
                OptimizerState &optimizerState
    ) override;

};

#endif //NN_OPTIMIZER_H
