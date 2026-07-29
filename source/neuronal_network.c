#include "headers/neuronal_network.h"

// sigmoid( ( weights[i] * neuron[i] ... + weights[n] * neuron[n] ) - bias )

// softmax activation function (e ** output[i])
// normalize softmax output[i] / num_of_outputs
// sigmoid activation function (1 / 1 + e ** -x)
// rectified linear activation function ( x <= 0 ? x : 0)

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

		float scale = input_parameters == 0 ? 1.f : sqrtf(2.0f / (perceptrons * input_parameters)); // Scale so weight do not explode
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
		float scale = sqrtf(2.0f / perceptrons); // Scale so weight do not explode
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
void propagateForward(Neuron *in, Neuron *out) {

	for (int x = 0; x < out->num_of_perceptrons; x++) {
		out->perceptrons[x].pre_activation = 0.f; // reset pre_activation!
		out->perceptrons[x].post_activation = 0.f; // reset post_activation!
		for (int y = 0; y < out->weights_dim_y; y++) {
			// dot Product between inputs and weights.
			out->perceptrons[x].pre_activation += in->perceptrons[y].pre_activation * out->weights[x][y];
		}
		out->perceptrons[x].pre_activation += out->perceptrons[x].bias;

		if (out->type != OUTPUT_LAYER) {
			out->perceptrons[x].post_activation = RelUActivation(out->perceptrons[x].pre_activation);
		}
	}
	if (out->type == OUTPUT_LAYER) {
		softmaxActivation(out);
	}
}
void softmaxActivation(Neuron *out) {
	float max_out = INT32_MIN;
	//float eulersNum = 2.71828182846f;

	//for (int i = 0; i < out->num_of_perceptrons; i++) {
	//	printf("Pre-Activation %d: %f\n", i, out->perceptrons[i].pre_activation);
	//}

	for (int i = 0; i < out->num_of_perceptrons; i++) {
		if (out->perceptrons[i].pre_activation > max_out) {
			max_out = out->perceptrons[i].pre_activation;
		}
	}

	float expSum = 0.f;
	for (int i = 0; i < out->num_of_perceptrons; i++) {
		//float overflow = out->perceptrons[i].pre_activation - max_out; // avoid overflow!
		//out->perceptrons[i].post_activation = powf(eulersNum, overflow);

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
	//return num > 0 ? num : 0.f;
	return num > 0.0f ? num : 0.01f * num;
}
float RelUDerivative(const float num) {
	//return num > 0 ? 1.0f : 0;
	return num > 0.0f ? 1.0f : 0.01f * num;
}
float Sigmoid(const float num) {
	return 1.0f / (1.0f + expf(-num));
}
float SigmoidDerivative(const float num) {
	return num * (1.0f - num);
}
void propagateBackward(Neuron *out, Neuron *second, Neuron *first, Neuron *in, const int target[]) {
	float *delta_1 = calloc(out->num_of_perceptrons, sizeof(float));
	float *delta_2 = calloc(second->num_of_perceptrons, sizeof(float));
	float *delta_3 = calloc(first->num_of_perceptrons, sizeof(float));
	if (!delta_1 || !delta_2 || !delta_3) {
		fprintf(stderr, "Could not allocate space for delta array! propagateBackward()");
		free(delta_1);
		free(delta_2);
		free(delta_3);
		return;
	}

	float error = 0.f;
	for (int i = 0; i < out->num_of_perceptrons; i++) {
		error = out->perceptrons[i].post_activation - target[i];
		delta_1[i] = error;
	}

	for (int x = 0; x < second->num_of_perceptrons; x++) {
		error = 0.f;
		for (int y = 0; y < out->num_of_perceptrons; y++) {
			error += delta_1[y] * out->weights[y][x];
		}
		delta_2[x] = error * RelUDerivative(second->perceptrons[x].pre_activation);
	}

	for (int x = 0; x < first->num_of_perceptrons; x++) {
		error = 0.f;
		for (int y = 0; y < second->num_of_perceptrons; y++) {
			error += delta_2[y] * second->weights[y][x];
		}
		delta_3[x] = error * RelUDerivative(first->perceptrons[x].pre_activation);
	}

	accumulateGradients(out, second, delta_1);
	accumulateGradients(second, first, delta_2);
	accumulateGradients(first, in, delta_3);

	free(delta_1);
	free(delta_2);
	free(delta_3);
}
void accumulateGradients(Neuron *in, Neuron *out, float *delta) {
	for (int x = 0; x < in->num_of_perceptrons; x++) {
		in->perceptrons[x].bias_acc += delta[x];

		for (int y = 0; y < in->weights_dim_y; y++) {
			in->weights_acc[x][y] += delta[x] * out->perceptrons[y].pre_activation; // Possible bug here. Maybe pre_activation must be used!
		}
	}
}
void updateBatchParameters(Neuron *n, const int batch_size, const float learning_rate) {
	for (int x = 0; x < n->num_of_perceptrons; x++) {		
		n->perceptrons[x].bias -= learning_rate * (n->perceptrons[x].bias_acc / (float)(batch_size));
		n->perceptrons[x].bias_acc = 0.f;

		for (int y = 0; y < n->weights_dim_y; y++) {
			//float final_weight_grad = n->weights_acc[x][y] / (float)(batch_size);
			//if (final_weight_grad > 0.5f)  final_weight_grad = 0.5f;
			//if (final_weight_grad < -0.5f) final_weight_grad = -0.5f;
			//n->weights[x][y] -= learning_rate * final_weight_grad;

			n->weights[x][y] -= learning_rate * (n->weights_acc[x][y] / (float)(batch_size));

			n->weights_acc[x][y] = 0.f;
		}
	}
}
float evaluateTestSet(Neuron *in, Neuron *first, Neuron *second, Neuron *out, Dataset *dt) {
	int correct_predictions = 0;
	for (int i = 0; i < dt->total_digits; i++) {
		initInputNeuron(in, &dt->digits[i]);
		propagateForward(in, first);
		propagateForward(first, second);
		propagateForward(second, out);

		int predicted_output = getNetworkPrediction(out);
		if (predicted_output == dt->digits[i].value) {
			correct_predictions++;
		}
	}
	printf("Neuronal Network correct predictions: %d\n", correct_predictions);
	return (correct_predictions / (float)(dt->total_digits)) * 100.f;
}
int getNetworkPrediction(Neuron *n) {
	int max_index = 0;
	float max_value = INT_MIN;
	for (int i = 0; i < n->num_of_perceptrons; i++) {
		if (n->perceptrons[i].post_activation > max_value) {
			max_value = n->perceptrons[i].post_activation;
			max_index = i;
		}
	}
	return max_index;
}
int identifyDigit(Neuron *in, Neuron *first, Neuron *second, Neuron *out, Digit *dg) {
	initInputNeuron(in, dg);
	propagateForward(in, first);
	propagateForward(first, second);
	propagateForward(second, out);

	return getNetworkPrediction(out);
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