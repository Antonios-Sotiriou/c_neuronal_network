#include "headers/neuronal_network.h"

// sigmoid( ( weights[i] * neuron[i] ... + weights[n] * neuron[n] ) - bias )

// softmax activation function (e ** output[i])
// normalize softmax output[i] / num_of_outputs
// sigmoid activation function (1 / 1 + e ** -x)
// rectified linear activation function ( x <= 0 ? x : 0)

void createNeuronalNetwork(NeuronalNetwork *nt, const int neuronLayers) {
	nt->neurons = calloc(neuronLayers, sizeof(Neuron));
	nt->total_neurons = neuronLayers;
}
void initNeuron(Neuron *n, const int type, const int input_parameters, const int perceptrons) {
	srand(time(0));

	n->type = type;
	n->weights_dim_x = perceptrons;
	n->weights_dim_y = input_parameters;
	if (n->type != INPUT_LAYER) {
		n->weights = malloc(sizeof(double) * n->weights_dim_x);
		n->weights_acc = malloc(sizeof(double) * n->weights_dim_x);
		if (!n->weights) {
			return;
		} else {
			for (int i = 0; i < n->weights_dim_x; i++) {
				n->weights[i] = malloc(sizeof(float) * n->weights_dim_y);
				n->weights_acc[i] = calloc(n->weights_dim_y, sizeof(float));
				if (!n->weights[i]) {
					fprintf(stdout, "Failed to allocate memory for n->weights[%d]\n", i);
					return;
				}
			}
		}

		float scale = input_parameters == 0 ? 1.f : sqrtf(1.0f / (perceptrons * input_parameters)); // Scale so weight do not explode
		for (int x = 0; x < n->weights_dim_x; x++) {
			for (int y = 0; y < n->weights_dim_y; y++) {
				n->weights[x][y] = ((rand() / (float)RAND_MAX) * 2.f - 1.f) * scale;
			}
		}
	}

	n->num_of_perceptrons = perceptrons;
	n->perceptrons = malloc(sizeof(Perceptron) * perceptrons);
	if (!n->perceptrons) {
		return;
	} else {
		float scale = sqrtf(1.0f / perceptrons); // Scale so weight do not explode
		for (int i = 0; i < perceptrons; i++) {
			n->perceptrons[i].bias = ((rand() / (float)RAND_MAX) * 2.f - 1.f) * scale;
			n->perceptrons[i].bias_acc = 0.f;
			n->perceptrons[i].post_activation = 0.f;
			n->perceptrons[i].pre_activation = 0.f;
		}
	}
}
void initInputNeuron(Neuron *n, Digit *dg) {
	for (int i = 0; i < n->num_of_perceptrons; i++) {
		n->perceptrons[i].pre_activation = dg->pixels[i] / 255.f;
	}
}
void propagateForward(NeuronalNetwork *nt) {
	int next = 1;
	for (int i = 0; i < nt->total_neurons; i++) {

		for (int x = 0; x < nt->neurons[next].num_of_perceptrons; x++) {
			nt->neurons[next].perceptrons[x].pre_activation = 0.f; // reset pre_activation!
			nt->neurons[next].perceptrons[x].post_activation = 0.f; // reset post_activation!
			for (int y = 0; y < nt->neurons[next].weights_dim_y; y++) {
				// dot Product between inputs and weights.
				nt->neurons[next].perceptrons[x].pre_activation += nt->neurons[i].perceptrons[y].pre_activation * nt->neurons[next].weights[x][y];
			}
			nt->neurons[next].perceptrons[x].pre_activation += nt->neurons[next].perceptrons[x].bias;

			if (nt->neurons[next].type != OUTPUT_LAYER) {
				nt->neurons[next].perceptrons[x].post_activation = RelUActivation(nt->neurons[next].perceptrons[x].pre_activation);
			}
		}

		if (i == 2) {
			break;
		}
		next++;
	}

	softmaxActivation(&nt->neurons[outputNeuron]);
}
void softmaxActivation(Neuron *out) {
	float max_out = INT32_MIN;

	for (int i = 0; i < out->num_of_perceptrons; i++) {
		if (out->perceptrons[i].pre_activation > max_out) {
			max_out = out->perceptrons[i].pre_activation;
		}
	}

	float expSum = 0.f;
	for (int i = 0; i < out->num_of_perceptrons; i++) {
		out->perceptrons[i].post_activation = expf(out->perceptrons[i].pre_activation - max_out);
		expSum += out->perceptrons[i].post_activation;
	}

	for (int i = 0; i < out->num_of_perceptrons; i++) {
		out->perceptrons[i].post_activation /= expSum; // Normalize!
	}
}
float lossEntropy(Neuron *out, const int target[]) {
	float loss = 0.f;
	for (int i = 0; i < out->num_of_perceptrons; i++) {
		// make sure we dont get a Zero to pass to log();
		float temp_posibillity = out->perceptrons[i].post_activation;

		if (temp_posibillity < 1e-7) {
			temp_posibillity = (float)(1e-7);
		} else if (temp_posibillity > 1 - (1e-7)) {
			temp_posibillity = (float)(1 - (1e-7));
		}
		loss += -logf(temp_posibillity) * target[i];
	}
	loss /= out->num_of_perceptrons;

	return loss;
}
float RelUActivation(const float num) {
	return num > 0 ? num : 0.f;
	//return num > 0.0f ? num : 0.01f * num;
}
float RelUDerivative(const float num) {
	return num > 0 ? 1.0f : 0;
	//return num > 0.0f ? 1.0f : 0.01f * num;
}
float Sigmoid(const float num) {
	return 1.0f / (1.0f + expf(-num));
}
float SigmoidDerivative(const float num) {
	return num * (1.0f - num);
}
void propagateBackward(NeuronalNetwork *nt, const int target[]) {
	float **delta = calloc(nt->total_neurons - 1, sizeof(double));
	if (!delta) {
		fprintf(stderr, "Could not allocate space for delta array! propagateBackward()");
		exit(-1);
	}
	int delta_index = 0;

	for (int active_neuron = nt->total_neurons - 1; active_neuron > 0; active_neuron--) {
		float error = 0.f;
		delta[delta_index] = calloc(nt->neurons[active_neuron].num_of_perceptrons, sizeof(float));
		if (!delta[delta_index]) {
			fprintf(stderr, "Could not allocate space for delta[delta_index]! propagateBackward()");
			exit(-1);
		}

		if (active_neuron == nt->total_neurons - 1) {
			for (int x = 0; x < nt->neurons[active_neuron].num_of_perceptrons; x++) {
				error = nt->neurons[active_neuron].perceptrons[x].post_activation - target[x];
				delta[delta_index][x] = error;
			}
		} else {
			int previous_neuron = active_neuron + 1; // Plus 1 because we count in reverse, namly descending.
			int previous_delta = delta_index - 1;
			for (int x = 0; x < nt->neurons[active_neuron].num_of_perceptrons; x++) {
				error = 0.f;
				for (int y = 0; y < nt->neurons[previous_neuron].num_of_perceptrons; y++) {
					error += delta[previous_delta][y] * nt->neurons[previous_neuron].weights[y][x];
				}
				delta[delta_index][x] = error * RelUDerivative(nt->neurons[active_neuron].perceptrons[x].pre_activation);
			}
		}
		delta_index++;
	}

	accumulateGradients(nt, delta);

	for (int i = 0; i < nt->total_neurons - 1; i++) {
		free(delta[i]);
	}
	free(delta);
}
void accumulateGradients(NeuronalNetwork *nt, float **delta) {
	int delta_index = 0;
	for (int active_neuron = nt->total_neurons - 1; active_neuron > 0; active_neuron--) {
		int next_neuron = active_neuron - 1;

		for (int x = 0; x < nt->neurons[active_neuron].num_of_perceptrons; x++) {

			nt->neurons[active_neuron].perceptrons[x].bias_acc += delta[delta_index][x];

			for (int y = 0; y < nt->neurons[active_neuron].weights_dim_y; y++) {
				nt->neurons[active_neuron].weights_acc[x][y] += delta[delta_index][x] * nt->neurons[next_neuron].perceptrons[y].pre_activation; // Possible bug here. Maybe pre_activation must be used!
			}
		}
		delta_index++;
	}
}
void updateNetworkBatchParameters(NeuronalNetwork *nt, const int batch_size, const float learning_rate) {
	for (int active_neuron = nt->total_neurons - 1; active_neuron > 0; active_neuron--) {
		updateNeuronBatchParameters(&nt->neurons[active_neuron], batch_size, learning_rate);
	}
}
void updateNeuronBatchParameters(Neuron *n, const int batch_size, const float learning_rate) {
	for (int x = 0; x < n->num_of_perceptrons; x++) {		
		n->perceptrons[x].bias -= learning_rate * (n->perceptrons[x].bias_acc / (float)(batch_size));
		n->perceptrons[x].bias_acc = 0.f;

		for (int y = 0; y < n->weights_dim_y; y++) {
			n->weights[x][y] -= learning_rate * (n->weights_acc[x][y] / (float)(batch_size));
			n->weights_acc[x][y] = 0.f;
		}
	}
}
float evaluateTestSet(NeuronalNetwork *nt, Dataset *dt) {
	int correct_predictions = 0;
	for (int i = 0; i < dt->total_digits; i++) {
		initInputNeuron(&nt->neurons[inputNeuron], &dt->digits[i]);
		propagateForward(nt);

		int predicted_output = getNetworkPrediction(nt);
		if (predicted_output == dt->digits[i].value) {
			correct_predictions++;
		}
	}

	return (correct_predictions / (float)(dt->total_digits)) * 100.f;
}
int getNetworkPrediction(NeuronalNetwork *nt) {
	int max_index = 0;
	float max_value = INT_MIN;
	for (int i = 0; i < nt->neurons[outputNeuron].num_of_perceptrons; i++) {
		if (nt->neurons[outputNeuron].perceptrons[i].post_activation > max_value) {
			max_value = nt->neurons[outputNeuron].perceptrons[i].post_activation;
			max_index = i;
		}
	}
	return max_index;
}
int identifyDigit(NeuronalNetwork *nt, Digit *dg) {
	initInputNeuron(&nt->neurons[inputNeuron], dg);
	propagateForward(nt);

	return getNetworkPrediction(nt);
}
void printNeuronWeights(Neuron *n) {
	for (int x = 0; x < n->weights_dim_x; x++) {
		for (int y = 0; y < n->weights_dim_y; y++) {
			printf("n->weights[%d][%d]: %f\n", x, y, n->weights[x][y]);
			printf("n->weights_acc[%d][%d]: %f\n", x, y, n->weights_acc[x][y]);
		}
	}
}
void printNeuron(Neuron *n) {
	for (int i = 0; i < n->num_of_perceptrons; i++) {
		printf("Neuron cell %d    bias: %f    bias_acc: %f    pre_activation: %f    post_activation: %f\n", i, n->perceptrons[i].bias, n->perceptrons[i].bias_acc, n->perceptrons[i].pre_activation, n->perceptrons[i].post_activation);
	}
}
void freeNeuron(Neuron *n) {
	for (int i = 0; i < n->num_of_perceptrons; i++) {
		if (n->type != INPUT_LAYER) {
			free(n->weights[i]);
			free(n->weights_acc[i]);
		}
	}
	free(n->weights);
	free(n->weights_acc);
	free(n->perceptrons);
}
void freeNeuronalNetwork(NeuronalNetwork *nt) {
	for (int i = 0; i < nt->total_neurons; i++) {
		freeNeuron(&nt->neurons[i]);
	}
	free(nt->neurons);
}