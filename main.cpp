#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>
#include <memory>

#include "Layer.h"
#include "AnyMovable.h"
#include "Optimizer.h"
#include "DataHandler.h"
#include "Loss.h"
#include "NeuralNetwork.h"

using namespace NeuralNet;


int argmax(const Vector &vec)
{
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return maxIndex;
}


Vector labelToOneHot(int label)
{
    Vector oneHot = Vector::Zero(
            10);
    if (label >= 0 && label < 10)
    {
        oneHot(label) = 1.0;
    } else
    {
        std::cerr << "Label out of bounds"
                  << std::endl;
    }
    return oneHot;
}

double calculateMean(const std::vector<double> &numbers)
{
    if (numbers.empty()) return 0.0;  // Return 0 if the vector is empty to avoid division by zero

    double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    return sum / numbers.size();
}

Vector calculateColwiseMean(const Eigen::MatrixXd &matrix)
{
    if (matrix.rows() == 0)
        return Vector();  // Return an empty vector if the matrix is empty

    // Compute the mean of each column
    Vector mean = matrix.colwise().mean();

    return mean;
}

int main()
{
//    CAnyLayer layer1 = LinearLayer(1, 1);
    CAnyLayer layer1 = LinearLayer(784, 300);
//    CAnyLayer activation1 = SigmoidLayer(300);
//    CAnyLayer layer2 = LinearLayer(300, 100);
//    CAnyLayer activation2 = SigmoidLayer(100);
//    CAnyLayer layer3 = LinearLayer(100, 10);
//    CAnyLayer activation3 = SoftmaxLayer();
//
//    std::vector<CAnyLayer> layers;
//
//    layer1;
return 0;}

