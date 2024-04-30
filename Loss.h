//
// Created by Kirill Belyakov on 4/22/24.
//

#ifndef NN_FROM_SCRATCH_LOSS_H
#define NN_FROM_SCRATCH_LOSS_H

#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>
#include "AnyMovable.h"


namespace NeuralNet
{
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;


template<class TBase>
class ILoss : public TBase
{
public:
    virtual Vector computeLoss(const Matrix &predictions, const Matrix &targets) = 0;

    virtual Matrix
    computeGradient(const Matrix &predictions, const Matrix &targets) = 0;

};


template<class TBase, class TObject>
class CLossImpl : public TBase
{
    using CBase = TBase;
public:
    using CBase::CBase;

    Vector computeLoss(const Matrix &predictions, const Matrix &targets) override
    {
        return CBase::Object().computeLoss(predictions, targets);
    }

    Matrix computeGradient(const Matrix &predictions, const Matrix &targets) override
    {
        return CBase::Object().computeGradient(predictions, targets);
    }
};

class CAnyLoss : public NSLibrary::CAnyMovable<ILoss, CLossImpl>
{
    using CBase = NSLibrary::CAnyMovable<ILoss, CLossImpl>;
public:
    using CBase::CBase;
};


class MSELoss
{
public:
    Vector computeLoss(const Matrix &predictions, const Matrix &targets);

    Matrix computeGradient(const Matrix &predictions, const Matrix &targets);
};


class BinaryCrossEntropyLoss
{
public:
    Vector computeLoss(const Matrix &predictions, const Matrix &targets);

    Matrix computeGradient(const Matrix &predictions, const Matrix &targets);
};
}
#endif //NN_FROM_SCRATCH_LOSS_H
