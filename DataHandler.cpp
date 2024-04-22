//
// Created by Kirill Belyakov on 4/20/24.
//

#include "DataHandler.h"

void NeuralNet::DataHandler::skipFirstLine(std::ifstream &file)
{
    std::string line;
    getline(file, line);
}

void NeuralNet::DataHandler::readData(const std::string &filename)
{
    std::ifstream file(filename);
    skipFirstLine(file);

    int numCols = 0;
    int numRows = 0;
    std::streampos oldPos = file.tellg();

    std::string line;
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


    features_.resize(numRows, numCols);
    labels_.reserve(numRows);

    int rowIndex = 0;
    std::string cell;
    while (getline(file, line))
    {
        std::stringstream lineStream(line);

        getline(lineStream, cell, ',');
        labels_.push_back(stoi(cell));

        int colIndex = 0;
        while (getline(lineStream, cell, ','))
        {
            features_(rowIndex, colIndex++) = std::stod(cell);
        }
        rowIndex++;
    }
}
