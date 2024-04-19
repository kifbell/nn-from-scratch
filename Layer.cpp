//
// Created by Kirill Belyakov on 4/18/24.
//

#include "Layer.h"
#include "Optimizer.h"


//class LinearLayer : public Layer
//{
//private:
//    Eigen::MatrixXd z_cache;  // Cache to store input for backpropagation
//
//public:
//    Eigen::MatrixXd weights;  // Matrix for weights
//    Eigen::VectorXd bias;     // Vector for biases
//    // Constructor to initialize weights and biases
//    LinearLayer(int input_size, int output_size) : Layer()
//    {
//        // Random initialization of weights and biases for demonstration purposes
//        weights = Eigen::MatrixXd::Random(output_size, input_size);
//        bias = Eigen::VectorXd::Random(output_size);
//
//        optimizerState = nullptr;
//    }
//
//    // Forward pass which calculates Wx + b
//    Eigen::VectorXd pass_forward(const Eigen::VectorXd &input)
//    {
//        z_cache = input;  // Cache the input (z) for use in backpropagation
//        return weights * input + bias;  // Wx + b
//    }
//
//    // Backpropagation method that updates weights and biases
////    Eigen::VectorXd backprop(const Eigen::VectorXd &u)
////    {
//    // Convert dsigma vector to a diagonal matrix
////        Eigen::MatrixXd Dsigma = dsigma.asDiagonal();
//
//    // Calculate gradients
////        Eigen::MatrixXd deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
////        Eigen::VectorXd deltaB = u;  // (dσ)^T * u^T
//
//    // Update weights and biases
////        weights -= learning_rate * deltaA.transpose();  // Update weights
////        bias -= learning_rate * deltaB;  // Update biases
//
//    // Calculate and return the gradient to pass to the previous layer
////        Eigen::VectorXd u_bar = weights.transpose() * u;  // u * dσ * A
////
////        return u_bar;
////    }
//
////    template<class O>
////    Eigen::VectorXd backprop(O &optimizer, const Eigen::MatrixXd &u)
//    Eigen::VectorXd backprop(Optimizer &optimizer, const Eigen::VectorXd &u)
//    {
////        optimizer.update(*this, u);
////        optimizer.update(this->optimizerState, u);
//
//
////        Eigen::MatrixXd deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
////        Eigen::VectorXd deltaB = u;  // (dσ)^T * u^T
//
//        Eigen::MatrixXd deltaA = u * z_cache.transpose();  // (dσ)^T * u^T * z^T
//        Eigen::VectorXd deltaB = u;  // (dσ)^T * u^T
//
//
//        optimizer.update(
////                *this
//                this->weights,
//                this->bias,
//                deltaA,
//                deltaB,
//                this->optimizerState
//        );
//
//        Eigen::VectorXd u_bar = weights.transpose() * u;  // u * dσ * A
//
//        return u_bar;
//    }
//
//    Eigen::MatrixXd &getWeights()
//    { return weights; }
//
//    Eigen::VectorXd &getBiases()
//    { return bias; }
//
//    OptimizerState *getState() const
//    { return optimizerState.get(); }
//
//};


class ReLULayer : public Layer
{
private:

    Eigen::MatrixXd dsigma;  // Matrix to store the derivatives


public:

    ReLULayer(int size)
    {
        // Initialize dsigma as a diagonal matrix of size 'size'.
        // The initial value assumes the derivative of ReLU for all positive inputs (1).
        // This will be recalculated during the forward pass.
        dsigma = Eigen::MatrixXd::Identity(size, size);
    }

    // The forward pass applies the ReLU function and calculates the correct dsigma
    Eigen::MatrixXd pass_forward(const Eigen::VectorXd &input)
    {
        // Calculate the output using ReLU activation
        Eigen::MatrixXd output = input.unaryExpr(
                [](double elem) { return std::max(0.0, elem); });

        // Recalculate dsigma based on the input
        dsigma = input.unaryExpr(
                [](double elem) { return elem > 0 ? 1.0 : 0.0; }).asDiagonal();

        return output;
    }

    Eigen::MatrixXd backprop(const Eigen::VectorXd &u)
    {
        // Apply the stored dsigma to the incoming gradient
        return dsigma * u;
    }

    OptimizerState *getState() const
    { return optimizerState.get(); }
};


