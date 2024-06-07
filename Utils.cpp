#include "Utils.h"

namespace NeuralNet
{
int argmax(const Vector &vec)
{
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return maxIndex;
}

Vector labelToOneHot(int label, const int numClasses)
{
    Vector oneHot = Vector::Zero(numClasses);
    if (label >= 0 && label < numClasses)
    {
        oneHot(label) = 1.0;
    } else
    {
        std::cerr << "Label out of bounds" << std::endl;
    }
    return oneHot;
}


Matrix labelsToOneHot(const Vector &labels, const int numClasses)
{
    int batchSize = labels.size();
    Matrix targetVectors(numClasses, batchSize);
    int colIndex = 0;
    for (int label: labels)
    {
        Vector targetVector = labelToOneHot(label, numClasses);
        targetVectors.col(colIndex++) = targetVector;
    }
    return targetVectors;
}

Vector calculateColwiseMean(const Eigen::MatrixXd &matrix)
{
    if (matrix.rows() == 0)
        return Vector();

    Vector mean = matrix.colwise().mean();

    return mean;
}


int countCorrectPredictions(const Matrix &predictions, const Matrix &targetVectors)
{
    if (predictions.cols() != targetVectors.cols() ||
        predictions.rows() != targetVectors.rows())
    {
        throw std::invalid_argument(
                "Predictions and targetVectors must have the same dimensions.");
    }

    int correctCount = 0;

    for (int i = 0; i < predictions.cols(); ++i)
    {
        Eigen::Index predictedClass;
        predictions.col(i).maxCoeff(&predictedClass);

        if (targetVectors(predictedClass, i) == 1)
        {
            ++correctCount;
        }
    }
    return correctCount;
}

Vector calculateRowwiseMean(const Eigen::MatrixXd &matrix)
{
    if (matrix.cols() == 0)
        return Vector();
    Vector mean = matrix.rowwise().mean();

    return mean;
}
}// namespace NeuralNet
