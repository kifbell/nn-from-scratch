#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include "Optimizer.h"
#include <map>
#include <string>


namespace NeuralNet
{

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;


class Layer
{
public:
    virtual Vector passForward(const Vector &input) = 0;

    virtual Vector
    backprop(Optimizer &optimizer, const Vector &u) = 0;

};

class LinearLayer : public Layer
{
private:
    Matrix z_cache;  // Cache to store input for backpropagation

public:
    Matrix weights_;
    Vector bias_;

    std::map<std::string, Matrix> optimizerState_;

    LinearLayer(int input_size, int output_size)
            : weights_(Matrix::Random(output_size, input_size)),
              bias_(Vector::Random(output_size))
    {}

    Vector passForward(const Vector &input) override;

    Vector backprop(Optimizer &optimizer, const Vector &u) override;
};


class ReLULayer : public Layer
{
private:
    Matrix dsigma;  // Matrix to store the derivatives
public:
    std::map<std::string, Matrix> optimizerState;

    explicit ReLULayer(int size) : dsigma(size, size)
    {}

    Vector passForward(const Vector &input) override;

    Vector backprop(Optimizer &optimizer, const Vector &u) override;

};


class SigmoidLayer : public Layer
{
private:
    Matrix dsigma;  // Matrix to store the derivatives
public:
    std::map<std::string, Matrix> optimizerState;

    explicit SigmoidLayer(int size) : dsigma(size, size)
    {}

    Vector passForward(const Vector &input) override;

    Vector backprop(Optimizer &optimizer, const Vector &u) override;
};


class SoftmaxLayer : public Layer
{
private:
    Vector lastInput_;

public:
    SoftmaxLayer()
    {}

    ~SoftmaxLayer()
    {}

    Vector passForward(const Vector &input) override;

    Matrix getJacobian(Vector& output);

    Vector backprop(Optimizer &optimizer, const Vector &u) override;
};
}


#endif //LAYER_HPP
