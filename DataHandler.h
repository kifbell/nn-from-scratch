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
class DataHandler
{
private:
    std::vector<int> labels;
    Eigen::MatrixXd features;
    size_t currentBatchIndex = 0;

public:
    DataHandler()
    {}

    void readData(const std::string &filename)
    {
        std::ifstream file(filename);
        std::string line, cell;

        getline(file, line);


        int numCols = 0;
        int numRows = 0;
        std::streampos oldPos = file.tellg();

        while (getline(file, line))
        {
            numRows++;
            if (numRows == 1)
            {

                numCols = std::count(line.begin(), line.end(), ',');
            }
        }


        file.clear();
        file.seekg(oldPos);


        features.resize(numRows, numCols);
        labels.reserve(numRows);

        int rowIndex = 0;
        while (getline(file, line))
        {
            std::stringstream lineStream(line);

            getline(lineStream, cell, ',');
            labels.push_back(stoi(cell));

            int colIndex = 0;
            while (getline(lineStream, cell, ','))
            {
                features(rowIndex, colIndex++) = std::stod(cell);
            }
            rowIndex++;
        }
        file.close();
    }

    std::pair<std::vector<int>, Eigen::MatrixXd> getRandomBatch(int batchSize)
    {
        std::vector<int> batchLabels;
        Eigen::MatrixXd batchFeatures(batchSize, features.cols());

        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<> distr(0, labels.size() - 1);

        for (int i = 0; i < batchSize; i++)
        {
            int index = distr(eng);
            batchLabels.push_back(labels[index]);
            batchFeatures.row(i) = features.row(index);
        }

        return {batchLabels, batchFeatures};
    }

    std::pair<std::vector<int>, Eigen::MatrixXd> getNextBatch(int batchSize)
    {
        int numSamples = std::min(batchSize,
                                  static_cast<int>(labels.size() - currentBatchIndex));
        Eigen::MatrixXd batchFeatures(numSamples, features.cols());
        std::vector<int> batchLabels(numSamples);

        for (int i = 0; i < numSamples; i++, currentBatchIndex++)
        {
            batchLabels[i] = labels[currentBatchIndex];
            batchFeatures.row(i) = features.row(currentBatchIndex);
        }

        return {batchLabels, batchFeatures};
    }

    void resetBatchIndex()
    {
        currentBatchIndex = 0;
    }

    int getNumberOfSamples() const
    {
        return labels.size();
    }

    int getNumberOfColumns() const
    {
        return features.cols();
    }
};
}

#endif // NN_FROM_SCRATCH_DATAHANDLER_H
