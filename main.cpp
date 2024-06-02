#include "NeuralNetwork.h"
#include "Except.h"


using namespace NeuralNet;


int main()
{
    try
    {
        NeuralNetwork::runAllTests();
    } catch (...)
    {
        except::react();
    }
    return 0;
}