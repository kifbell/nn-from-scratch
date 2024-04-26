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

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;


class Random
{
public:
    int uniform_int(int a, int b)
    {
        std::uniform_int_distribution<> distr(a, b);
        return distr(engine_);
    }

private:
    static constexpr int kSeed = 42;
    std::mt19937 engine_{kSeed};
};


struct DataBatch
{
   Vector labels;
    Eigen::MatrixXd features;

    DataBatch(int batchSize, int numFeatures) : labels(batchSize),
                                                features(batchSize, numFeatures)
    {}
};


class DataHandler
{
private:
    Vector labels_;
    Matrix features_;
    size_t currentBatchIndex = 0;

public:
    DataHandler() = default;


    const DataBatch getRandomBatch(int batchSize)
    {
        static Random rnd;

        DataBatch batch(batchSize, features_.cols());

        std::uniform_int_distribution<> distr(0, labels_.size() - 1);
        std::vector<int> indices;
        for (int i = 0; i < batchSize; i++)
        {
            int index = rnd.uniform_int(0, labels_.size() - 1);
            indices.push_back(index);
        }

        batch.labels = labels_(indices);
        batch.features = features_(indices, Eigen::placeholders::all);

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
