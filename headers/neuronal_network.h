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
	Neuron neurons[1];
	int total_neurons;
} NeuronalNetwork;

void initNeuron(Neuron *n, const int type, const int input_parameters, const int perceptrons);
void initInputNeuron(Neuron *n, Digit *dg);
void propagateForward(Neuron *in, Neuron *out);
float RelUActivation(const float num);
float RelUDerivative(const float num);
float Sigmoid(const float num);
float SigmoidDerivative(const float num);
void softmaxActivation(Neuron *out);
float lossEntropy(Neuron *out, const int target[]);
void propagateBackward(Neuron *out, Neuron *second, Neuron *first, Neuron *in, const int target[]);
float *propagateBackwardHiddenLayer(Neuron *in, Neuron *next, float *prev_delta);
void accumulateGradients(Neuron *in, Neuron *out, float *delta);
void updateBatchParameters(Neuron *n, const int batch_size, const float learning_rate);
float evaluateTestSet(Neuron *in, Neuron *first, Neuron *second, Neuron *out, Dataset *dt);
int getNetworkPrediction(Neuron *n);
int identifyDigit(Neuron *in, Neuron *first, Neuron *second, Neuron *out, Digit *dg);
void printNeuronWeights(Neuron *n);
void printNeuron(Neuron *n);
void freeNeuron(Neuron *n);

#endif // !NEURONAL_NETWORK_H
