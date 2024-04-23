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


class ReLULayer
{
private:
    Matrix dsigma;  // Matrix to store the derivatives
    std::map<std::string, Matrix> optimizerState;
public:

    explicit ReLULayer(int size) : dsigma(size, size)
    {}

    Vector passForward(const Vector &input);

    Vector backprop(Optimizer &optimizer, const Vector &u);

};


class SigmoidLayer
{
private:
    Matrix dsigma;  // Matrix to store the derivatives
    std::map<std::string, Matrix> optimizerState;
public:

    explicit SigmoidLayer(int size) : dsigma(size, size)
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
}

#endif //LAYER_HPP
