#ifndef NN_FROM_SCRATCH_DATAHANDLER_H
#define NN_FROM_SCRATCH_DATAHANDLER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>

class DataHandler {
private:
    std::vector<int> labels; // Store labels in a vector
    std::vector<std::vector<double>> features; // Store features in a 2D vector
    size_t currentBatchIndex = 0; // Current index for ordered batch retrieval

public:
    DataHandler() {}

    // Method to read data from a CSV file
    void readData(const std::string& filename) {
        std::ifstream file(filename);
        std::string line, cell;

        getline(file, line); // skip fist line w/ the names of the columns
        while (getline(file, line)) {
            std::stringstream lineStream(line);
            std::vector<double> parsedRow;

            getline(lineStream, cell, ',');
            labels.push_back(stoi(cell)); // Store the label separately

            while (getline(lineStream, cell, ',')) {
                parsedRow.push_back(stod(cell)); // Convert string to integer
            }
            features.push_back(parsedRow);
        }
        file.close();
    }

    // Method to get random batches of data
    std::pair<std::vector<int>, std::vector<std::vector<double>>> getRandomBatch(int batchSize) {
        std::vector<int> batchLabels;
        std::vector<std::vector<double>> batchFeatures;

        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 eng(rd()); // Seed the generator
        std::uniform_int_distribution<> distr(0, labels.size() - 1); // Define the range

        for (int i = 0; i < batchSize; i++) {
            int index = distr(eng); // Generate a random index
            batchLabels.push_back(labels[index]); // Add the label at the index to the batch
            batchFeatures.push_back(features[index]); // Add the features at the index to the batch
        }

        return {batchLabels, batchFeatures};
    }

    // Method to get batches of data in order
    std::pair<std::vector<int>, std::vector<std::vector<double>>> getNextBatch(int batchSize) {
        std::vector<int> batchLabels;
        std::vector<std::vector<double>> batchFeatures;

        for (int i = 0; i < batchSize && currentBatchIndex < labels.size(); i++, currentBatchIndex++) {
            batchLabels.push_back(labels[currentBatchIndex]); // Add the label at the current index
            batchFeatures.push_back(features[currentBatchIndex]); // Add the features at the current index
        }

        return {batchLabels, batchFeatures};
    }

    void resetBatchIndex() {
        currentBatchIndex = 0;
    }

    int getNumberOfSamples() const {
        return labels.size();
    }
};

//int main() {
//    DataHandler handler;
//    handler.readData("data.csv"); // Specify the CSV file containing the data
//
//    // Get a random batch of 5 samples and print them
//    auto randomBatch = handler.getRandomBatch(5);
//    std::cout << "Random Batch:" << std::endl;
//    for (size_t i = 0; i < randomBatch.first.size(); i++) {
//        std::cout << "Label: " << randomBatch.first[i] << " Features: ";
//        for (int feature : randomBatch.second[i]) {
//            std::cout << feature << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    // Get an ordered batch of 5 samples and print them
//    auto orderedBatch = handler.getNextBatch(5);
//    std::cout << "\nOrdered Batch:" << std::endl;
//    for (size_t i = 0; i < orderedBatch.first.size(); i++) {
//        std::cout << "Label: " << orderedBatch.first[i] << " Features: ";
//        for (int feature : orderedBatch.second[i]) {
//            std::cout << feature << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    return 0;
//}

#endif //NN_FROM_SCRATCH_DATAHANDLER_H
