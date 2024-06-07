#include "Except.h"
#include "NeuralNetwork.h"


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