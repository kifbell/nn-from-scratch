#ifndef NN_FROM_SCRATCH_DATAHANDLER_H
#define NN_FROM_SCRATCH_DATAHANDLER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <Eigen/Dense>

namespace NeuralNet
{


struct DataBatch
{
    std::vector<int> labels;
    Eigen::MatrixXd features;

    DataBatch(int batchSize, int numFeatures) : labels(batchSize),
                                                features(batchSize, numFeatures)
    {}
};


class DataHandler
{
private:
    std::vector<int> labels_;
    Eigen::MatrixXd features_;
    size_t currentBatchIndex = 0;

public:
    DataHandler()=default;


    const DataBatch getRandomBatch(int batchSize)
    {

        DataBatch batch(batchSize, features_.cols());

        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<> distr(0, labels_.size() - 1);

        for (int i = 0; i < batchSize; i++)
        {
            int index = distr(eng);
            batch.labels[i] = labels_[index];
            batch.features.row(i) = features_.row(index);
        }

        return batch;
    }

    const DataBatch getNextBatch(int batchSize)
    {
        int numSamples = std::min(batchSize,
                                  static_cast<int>(labels_.size() - currentBatchIndex));
        DataBatch batch(batchSize, features_.cols());

        for (int i = 0; i < numSamples; i++, currentBatchIndex++)
        {
            batch.labels[i] = labels_[currentBatchIndex];
            batch.features.row(i) = features_.row(currentBatchIndex);
        }

        return batch;
    }

    void resetBatchIndex()
    {
        currentBatchIndex = 0;
    }

    int getNumberOfSamples() const
    {
        return labels_.size();
    }

    int getNumberOfColumns() const
    {
        return features_.cols();
    }

    void readData(const std::string &filename);

    static void skipFirstLine(std::ifstream &file);
};
}

#endif // NN_FROM_SCRATCH_DATAHANDLER_H
