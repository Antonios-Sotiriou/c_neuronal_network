#ifndef NEURONAL_NETWORK_H
#define NEURONAL_NETWORK_H 1

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Project specific headers
#ifndef DATASET_H
    #include "dataset.h"
#endif

#define INPUT_LAYER         1
#define FIRST_HIDDEN_LAYER  2
#define SECOND_HIDDEN_LAYER 3
#define OUTPUT_LAYER        4

enum { inputNeuron, firstNeuron, secondNeuron, outputNeuron };

typedef struct {
	float bias;
	float bias_acc;
	float pre_activation;
	float post_activation;
} Perceptron;

typedef struct {
	Perceptron *perceptrons;
	float **weights;
	float **weights_acc;
	int num_of_perceptrons, weights_dim_x, weights_dim_y;
	int type;
} Neuron;

typedef struct {
	Neuron *neurons;
	int total_neurons;
} NeuronalNetwork;

void createNeuronalNetwork(NeuronalNetwork *nt, const int neuronLayers);
void initNeuron(Neuron *n, const int type, const int input_parameters, const int perceptrons);
void initInputNeuron(Neuron *n, Digit *dg);
void propagateForward(NeuronalNetwork *nt);
float RelUActivation(const float num);
float RelUDerivative(const float num);
float Sigmoid(const float num);
float SigmoidDerivative(const float num);
void softmaxActivation(Neuron *out);
float lossEntropy(Neuron *out, const int target[]);
void propagateBackward(NeuronalNetwork *nt, const int target[]);
void accumulateGradients(NeuronalNetwork *nt, float **delta);
void updateNetworkBatchParameters(NeuronalNetwork *nt, const int batch_size, const float learning_rate);
void updateNeuronBatchParameters(Neuron *n, const int batch_size, const float learning_rate);
float evaluateTestSet(NeuronalNetwork *nt, Dataset *dt);
int getNetworkPrediction(Neuron *n);
int identifyDigit(NeuronalNetwork *nt, Digit *dg);
void printNeuronWeights(Neuron *n);
void printNeuron(Neuron *n);
void freeNeuron(Neuron *n);
void freeNeuronalNetwork(NeuronalNetwork *nt);

#endif // !NEURONAL_NETWORK_H
