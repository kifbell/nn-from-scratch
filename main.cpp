#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>

#include "Layer.h"
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


Matrix labelsToOneHot(const Vector &labels)
{
    int batchSize = labels.size();
    Matrix targetVectors(10, batchSize);
    int colIndex=0;
    for (int label: labels)
    {
        Vector targetVector = labelToOneHot(label);
//        std::cout << "targetVector: " << targetVector.transpose() << std::endl;

//        targetVectors << targetVector; // todo corrupted values
        targetVectors.col(colIndex++) = targetVector;

    }
//    std::cout << "targetVectors: " << targetVectors << std::endl;
//    std::cout << "targetVectors.shape " << targetVectors.rows() << ' ' << targetVectors.cols() << std::endl;
//    std::cout << "targetVectors: " << targetVectors << std::endl;

    return targetVectors;
}

Vector calculateColwiseMean(const Eigen::MatrixXd &matrix)
{
    if (matrix.rows() == 0)
        return Vector();  // Return an empty vector if the matrix is empty

    // Compute the mean of each column
    Vector mean = matrix.colwise().mean();

    return mean;
}


int countCorrectPredictions(const Matrix& predictions, const Matrix& targetVectors) {
    if (predictions.cols() != targetVectors.cols() || predictions.rows() != targetVectors.rows()) {
        throw std::invalid_argument("Predictions and targetVectors must have the same dimensions.");
    }

    int correctCount = 0;
    for (int i = 0; i < predictions.cols(); ++i) {
        // Find the index of the maximum value in each column of predictions
        int predictedClass = (int)predictions.col(i).maxCoeff(&predictedClass);

        // Check if the corresponding entry in the target vector is 1
        if (targetVectors(predictedClass, i) == 1) {
            ++correctCount;
        }
    }

    return correctCount;
}

Vector calculateRowwiseMean(const Eigen::MatrixXd &matrix)
{
    if (matrix.cols() == 0)
        return Vector();  // Return an empty vector if the matrix is empty

    // Compute the mean of each column
    Vector mean = matrix.rowwise().mean();

    return mean;
}


int trainNN(NeuralNetwork &nn,
            CAnyLoss &loss,
            Optimizer &optimizer,
            DataHandler &trainHandler,
            int batchSize,
            int epochs,
            double lr,
            double lrDecay
)
{
    int runsInEpoch = trainHandler.getNumberOfSamples() / batchSize;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double lrCrt= lr / (1 + lrDecay * epoch);
        optimizer.lrUpdate(lrCrt);
        for (int batchRun = 0; batchRun < runsInEpoch; batchRun++)
        {
            auto batch = trainHandler.getRandomBatch(batchSize);


//            std::cout << "batch: " << batch.features << std::endl;


//            batch.features.transpose() // todo check the shape

            Matrix prediction = nn.passForward(
                    batch.features.transpose() / 255);// todo check the shape
//            std::cout << "prediction.shape " << prediction.rows() << ' ' << prediction.cols() << std::endl;

            Matrix targetVectors = labelsToOneHot(batch.labels);
//            std::cout << "targetVectors.shape " << targetVectors.rows() << ' ' << targetVectors.cols() << std::endl;
//            std::cout << "targetVectors.shape " << targetVectors.rows() << ' ' << targetVectors.cols() << std::endl;
//            std::cout << "prediction " << prediction  << std::endl;

//            std::cout << "targetVectors " << targetVectors  << std::endl;
            Vector lossVector = loss->computeLoss(prediction, targetVectors);
//            std::cout << "current_loss: " << lossVector.transpose().mean() << std::endl;
            Matrix gradients = loss->computeGradient(prediction, targetVectors);
//            std::cout << "gradients: " << gradients.transpose() << std::endl;

            // batch loss history


            // step by mean batch gradient
            Vector gradientMean = calculateRowwiseMean(gradients);
//        std::cout << "gradientMean.transpose()" << gradientMean.transpose() << std::endl;
            nn.backprop(optimizer, gradientMean);


            int cnt = countCorrectPredictions(prediction, targetVectors);
            std::cout << "Epoch: " << epoch << " batchRun " << batchRun << '/'
                      << runsInEpoch << ", Loss: " << lossVector.transpose().mean()
                      << ", ration: " << cnt  << '/' << batchSize
                      << std::endl;
        }
        std::cout << "Epoch " << epoch << " ended." << std::endl;
    }
    return 1;
}

int main()
{

//    Matrix m = Matrix::Ones(2, 2);
//    std::cout << m <<std::endl;
//    Vector v = Vector::Ones(2);
//    Matrix mv(2, 2);
//    mv << v ,v ;
//    v.replicate(1, ) ;
//    std::cout << v;
//    std::cout << mv <<std::endl;
//    Matrix r  = m + mv;
//    std::cout << r <<std::endl;
//    return 0;
    DataHandler trainHandler;
    trainHandler.readData(
            "/Users/fuckingbell/programming/nn-from-scratch/data/mnist_test.csv");
    std::cout << "rows read by dataHandler: " << trainHandler.getNumberOfSamples()
              << std::endl;

    CAnyLayer layer1 = LinearLayer(784, 300);
    CAnyLayer activation1 = CwiseActivation::Sigmoid();
    CAnyLayer layer2 = LinearLayer(300, 100);
    CAnyLayer activation2 = CwiseActivation::Sigmoid();
    CAnyLayer layer3 = LinearLayer(100, 10);
    CAnyLayer activation3 = SoftmaxLayer();

    NeuralNetwork nn;
    nn.addLayer(std::move(layer1));
    nn.addLayer(std::move(activation1));
    nn.addLayer(std::move(layer2));
    nn.addLayer(std::move(activation2));
    nn.addLayer(std::move(layer3));
    nn.addLayer(std::move(activation3));


    CAnyLoss loss = MSELoss();

    int batchSize = 35;
    int epochs = 100;

    double lr = 0.1;
    double lrDecay = 0.9;
    MomentumOptimizer optimizer(lr, 0.9);  // Learning rate and momentum

    trainNN(nn, loss, optimizer, trainHandler, batchSize, epochs, lr, lrDecay);


    return 0;
    if (trainHandler.getNumberOfSamples() > 0)
    {
        std::cout << "Number of features per sample: "
                  << trainHandler.getNumberOfColumns()
                  << std::endl;
    }

    // Get a random batch of 5 samples and print them
    auto randomBatch = trainHandler.getRandomBatch(5);
    std::cout << "\nRandom Batch:" << std::endl;
    std::cout <<
              randomBatch.labels.size()
              << std::endl;
    for (size_t i = 0; i < randomBatch.labels.size(); i++)
    {
        std::cout << i << std::endl;
        std::cout << "Label: " << randomBatch.labels[i] << " Features: ";
        for (int j = 0; j < randomBatch.features.cols(); j++)
        {
            std::cout << randomBatch.features(i, j) << " ";
        }
        std::cout << std::endl;
    }

}

