#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include "OptimizerState.h"
#include "Optimizer.h"
#include <map>
#include <string>

class Layer
{
protected:
public:
    Layer()
    {};

    ~Layer()
    {};

    virtual Eigen::VectorXd passForward(const Eigen::VectorXd &input) = 0;

    virtual Eigen::VectorXd
    backprop(Optimizer &optimizer, const Eigen::VectorXd &u) = 0;

};

class LinearLayer : public Layer
{
private:
    Eigen::MatrixXd z_cache;  // Cache to store input for backpropagation

public:
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;

    std::map<std::string, Eigen::MatrixXd> optimizerState;

    LinearLayer(int input_size, int output_size) : Layer()
    {
        // Random initialization of weights and biases for demonstration purposes
        weights = Eigen::MatrixXd::Random(output_size, input_size);
        bias = Eigen::VectorXd::Random(output_size);
    }

    ~LinearLayer()
    {};

    Eigen::VectorXd passForward(const Eigen::VectorXd &input) override;

    Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u) override;
};


class ReLULayer : public Layer
{
private:
    Eigen::MatrixXd dsigma;  // Matrix to store the derivatives
public:
    std::map<std::string, Eigen::MatrixXd> optimizerState;

    ReLULayer(int size)
    {
        // Initialize dsigma as a diagonal matrix of size 'size'.
        // The initial value assumes the derivative of ReLU for all positive inputs (1).
        // This will be recalculated during the forward pass.
        dsigma = Eigen::MatrixXd::Identity(size, size);
    }

    ~ReLULayer()
    {};

    Eigen::VectorXd passForward(const Eigen::VectorXd &input) override;

    Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u) override;

};


class SigmoidLayer : public Layer
{
private:
    Eigen::MatrixXd dsigma;  // Matrix to store the derivatives
public:
    std::map<std::string, Eigen::MatrixXd> optimizerState;

    explicit SigmoidLayer(int size)
    {
        // Initialize dsigma as a diagonal matrix of size 'size'.
        // The matrix will be recalculated during each forward pass.
        dsigma = Eigen::MatrixXd::Zero(size, size);
    }

    virtual ~SigmoidLayer()
    {};

    Eigen::VectorXd passForward(const Eigen::VectorXd &input) override;

    Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u) override;
};


#endif //LAYER_HPP
