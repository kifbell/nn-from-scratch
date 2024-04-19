#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include "OptimizerState.h"
#include "Optimizer.h"
#include <map>


class Layer
{
protected:
public:
    Layer()
    {};

    ~Layer()
    {};

    virtual Eigen::VectorXd pass_forward(const Eigen::VectorXd &input) = 0;

    virtual Eigen::VectorXd
    backprop(Optimizer &optimizer, const Eigen::VectorXd &u) = 0;

//    virtual Eigen::MatrixXd &getWeights() = 0;
//    virtual Eigen::VectorXd &getBiases() = 0;
};


//class LinearLayer : public Layer {
//private:
//    Eigen::MatrixXd z_cache;  // Cache to store input for backpropagation
//    Eigen::MatrixXd weights;  // Matrix for weights
//    Eigen::VectorXd bias;     // Vector for biases
//    std::unique_ptr<OptimizerState> optimizerState;  // State managed by optimizer
//
//public:
//    LinearLayer(int input_size, int output_size);
//
//    virtual Eigen::VectorXd pass_forward(const Eigen::VectorXd &input)=0;
//
//    virtual Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u)=0;
//
//    virtual Eigen::MatrixXd &getWeights()=0;
//    virtual Eigen::VectorXd &getBiases()=0;
//
//    OptimizerState *getState() const;
//};

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


//        optimizerState = nullptr;
    }

    ~LinearLayer()
    {};


    // Forward pass which calculates Wx + b
    Eigen::VectorXd pass_forward(const Eigen::VectorXd &input)
    {
        z_cache = input;  // Cache the input (z) for use in backpropagation
        return weights * input + bias;  // Wx + b
    }

    // Backpropagation method that updates weights and biases
//    Eigen::VectorXd backprop(const Eigen::VectorXd &u)
//    {
    // Convert dsigma vector to a diagonal matrix
//        Eigen::MatrixXd Dsigma = dsigma.asDiagonal();

    // Calculate gradients
//        Eigen::MatrixXd deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
//        Eigen::VectorXd deltaB = u;  // (dσ)^T * u^T

    // Update weights and biases
//        weights -= learning_rate * deltaA.transpose();  // Update weights
//        bias -= learning_rate * deltaB;  // Update biases

    // Calculate and return the gradient to pass to the previous layer
//        Eigen::VectorXd u_bar = weights.transpose() * u;  // u * dσ * A
//
//        return u_bar;
//    }

//    template<class O>
//    Eigen::VectorXd backprop(O &optimizer, const Eigen::MatrixXd &u)
    Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u)
    {
        // Store original weights and biases for later comparison
        Eigen::MatrixXd originalWeights = this->weights;
        Eigen::VectorXd originalBiases = this->bias;

        Eigen::MatrixXd deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
        Eigen::VectorXd deltaB = u;  // (dσ)^T * u^T

        // Copy current weights and biases for later comparison
        Eigen::MatrixXd weightsBeforeUpdate = weights;
        Eigen::VectorXd biasesBeforeUpdate = bias;

        optimizer.update(
                this->weights,
                this->bias,
                deltaA,
                deltaB,
                this->optimizerState
        );


        assert(!this->weights.isApprox(originalWeights) &&
               "Weights should have changed after optimization.");
        assert(!this->bias.isApprox(originalBiases) &&
               "Biases should have changed after optimization.");


        Eigen::VectorXd u_bar = weights.transpose() * u;  // u * dσ * A

        return u_bar;
    }

    Eigen::MatrixXd &getWeights()
    { return weights; }

    Eigen::VectorXd &getBiases()
    { return bias; }

//    OptimizerState *getState() const
//    { return optimizerState.get(); }

};


#endif //LAYER_HPP

