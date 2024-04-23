#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include "Optimizer.h"
#include "AnyMovable.h"
#include <map>
#include <string>


namespace NeuralNet
{

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;


template<class TBase>
class ILayer : public TBase
{
public:
    virtual Vector passForward(const Vector &input) = 0;

    virtual Vector backprop(Optimizer &optimizer, const Vector &u) = 0;
};


template<class TBase, class TObject>
class CLayerImpl : public TBase
{
    using CBase = TBase;
public:
    using CBase::CBase;

    Vector passForward(const Vector &input) override
    {
        return CBase::Object().passForward(input);
    }

    Vector backprop(Optimizer &optimizer, const Vector &u) override
    {
        return CBase::Object().backprop(optimizer, u);
    }

};

class CAnyLayer : public NSLibrary::CAnyMovable<ILayer, CLayerImpl>
{
    using CBase = NSLibrary::CAnyMovable<ILayer, CLayerImpl>;
public:
    using CBase::CBase;

//    friend bool operator==(const CAnyLayer &, const CAnyLayer &)
//    {
//    }
};


class LinearLayer
{
private:
    Matrix z_cache;
    Matrix weights_;
    Vector bias_;
    std::map<std::string, Matrix> optimizerState_;
public:
    LinearLayer(int input_size, int output_size)
            : weights_(Matrix::Random(output_size, input_size)),
              bias_(Vector::Random(output_size))
    {}

    Vector passForward(const Vector &input);

    Vector backprop(Optimizer &optimizer, const Vector &u);
};

class SoftmaxLayer
{
private:
    Vector lastInput_;

public:
    SoftmaxLayer()
    {}

    ~SoftmaxLayer()
    {}

    Vector passForward(const Vector &input);

    Matrix getJacobian(Vector &output);

    Vector backprop(Optimizer &optimizer, const Vector &u);
};


class CwiseActivation
{
    using Function = std::function<double(double)>;
public:
    CwiseActivation(Function f0, Function f1) : f0_(std::move(f0)), f1_(std::move(f1))
    {}

    Vector passForward(const Vector &input)
    {
        return input.unaryExpr(f0_);
    }

    Vector backprop(Optimizer &, const Vector &u)
    {
        return u.unaryExpr(f1_);
    }

    static CwiseActivation ReLu()
    {
        return CwiseActivation(
                [](double x) { return x * (x > 0); },
                [](double x) { return x > 0; }
        );
    }

    static CwiseActivation Sigmoid()
    {
        return CwiseActivation(
                [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
                [](double x) { return x * (1 - x); }
        );
    }


private:
    Function f0_;
    Function f1_;
};
}

#endif //LAYER_HPP
