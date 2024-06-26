#ifndef LAYER_HPP
#define LAYER_HPP

#include "AnyMovable.h"
#include "Optimizer.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <map>
#include <string>


namespace NeuralNet
{

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;


template<class TBase>
class ILayer : public TBase
{

private:
    Matrix zCache_;

public:
    virtual Matrix passForward(const Matrix &input) = 0;

    virtual Vector backprop(Optimizer &optimizer, const Vector &u) = 0;
};


template<class TBase, class TObject>
class CLayerImpl : public TBase
{
    using CBase = TBase;
    Matrix zCache_;
    size_t inputCnt = 0;
    size_t batchSize_;

public:
    using CBase::CBase;

    Matrix passForward(const Matrix &input) override
    {

        return CBase::Object().passForward(input);
    }

    Vector backprop(Optimizer &optimizer, const Vector &u) override
    {
        //        std::cout << "CLayerImpl SoftmaxLayer zCache_.size() = "<< zCache_.size() << std::endl;

        return CBase::Object().backprop(optimizer, u);
    }
};

class CAnyLayer : public NSLibrary::CAnyMovable<ILayer, CLayerImpl>
{
    using CBase = NSLibrary::CAnyMovable<ILayer, CLayerImpl>;

public:
    using CBase::CBase;
};


class LinearLayer
{
private:
    Matrix zCache_;
    Matrix weights_;
    Vector bias_;
    OptimizerState optimizerState_;

public:
    LinearLayer(int input_size, int output_size, double scale)
        : weights_(Matrix::Random(output_size, input_size) / scale + Matrix::Random(output_size, input_size) / scale),
          bias_(Vector::Random(output_size) / scale + Vector::Random(output_size) / scale)
    {}

    Matrix passForward(const Matrix &input);

    Vector backprop(Optimizer &optimizer, const Vector &u);
};

class SoftmaxLayer
{
private:
    Matrix zCache_;
    size_t inputCnt = 0;

public:
    SoftmaxLayer()
    {}

    ~SoftmaxLayer()
    {}

    Matrix passForward(const Matrix &input);

    Matrix getJacobian(const Matrix &output);

    Vector backprop(Optimizer &optimizer, const Vector &u);
};


class CwiseActivation
{
    using Function = std::function<double(double)>;

private:
    Function f0_;
    Function f1_;
    Matrix zCache_;
    size_t inputCnt = 0;

public:
    CwiseActivation(Function f0, Function f1) : f0_(std::move(f0)), f1_(std::move(f1))
    {}

    Matrix passForward(const Matrix &input)
    {
        zCache_ = input;
        return input.unaryExpr(f0_);
    }

    Matrix backprop(Optimizer &, const Vector &u)
    {
        int batchSize = zCache_.cols();
        Matrix jacobian = passForward(zCache_).unaryExpr(
                                                      f1_)
                                  .rowwise()
                                  .mean()
                                  .asDiagonal();
        return jacobian * u;
    }

    static CwiseActivation ReLu()
    {
        return CwiseActivation(
                [](double x) { return x * (x > 0); },
                [](double y) { return y > 0; });
    }

    static CwiseActivation Sigmoid()
    {
        return CwiseActivation(
                [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
                [](double y) { return y * (1 - y); });
    }

    static CwiseActivation Tanh()
    {
        return CwiseActivation(
                [](double x) { return std::tanh(x); },
                [](double y) { return 1.0 - std::pow(std::tanh(y), 2); });
    }
};
}// namespace NeuralNet

#endif//LAYER_HPP
