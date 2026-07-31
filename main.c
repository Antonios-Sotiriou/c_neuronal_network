#include "headers/main.h"

int main(int argv, char *argc[]) {

	Dataset mnist = { 0 };

	readDataset(&mnist, "mnist_train.csv");
	if (!mnist.digits) {
		fprintf(stderr, "(mnist.digits == 0). No data have been loaded!");
		return -1;
	}

	NeuronalNetwork net = { 0 };
	createNeuronalNetwork(&net, 4);
	initNeuron(&net.neurons[inputNeuron], INPUT_LAYER, 0, mnist.pixels_pro_digit);
	initNeuron(&net.neurons[firstNeuron], FIRST_HIDDEN_LAYER, net.neurons[inputNeuron].num_of_perceptrons, 40);
	initNeuron(&net.neurons[secondNeuron], SECOND_HIDDEN_LAYER, net.neurons[firstNeuron].num_of_perceptrons, 40);
	initNeuron(&net.neurons[outputNeuron], OUTPUT_LAYER, net.neurons[secondNeuron].num_of_perceptrons, 10);

	int epochs = 30;
	int batch_size = 16;
	float learning_rate = 0.002;

	printf("Neuronal Network {\n");
	printf("    Epochs              : %d\n", epochs);
	printf("    batch Size          : %d\n", batch_size);
	printf("    Learning Rate       : %f\n", learning_rate);
	printf("    Input Perceptrons   : %d\n", net.neurons[inputNeuron].num_of_perceptrons);
	printf("    Layer 1 Perceptrons : %d\n", net.neurons[firstNeuron].num_of_perceptrons);
	printf("    Layer 2 Perceptrons : %d\n", net.neurons[secondNeuron].num_of_perceptrons);
	printf("    Output Perceptrons  : %d\n", net.neurons[outputNeuron].num_of_perceptrons);
	printf("    Samples {\n");

	for (int epoch = 0; epoch < epochs; epoch++) {
		float epoch_loss_sum = 0.0f;
		int correct_predictions = 0;
		int count_samples = 0;

		for (int i = 0; i < mnist.total_digits; i++) {

			initInputNeuron(&net.neurons[inputNeuron], &mnist.digits[i]);
			propagateForward(&net);

			int target_output[10] = { 0 };
			target_output[mnist.digits[i].value] = 1;
			float loss = lossEntropy(&net.neurons[outputNeuron], target_output);
			epoch_loss_sum += loss;

			//printf("Computed loss: %f\n", loss);
			int prediction = getNetworkPrediction(&net);
			if (prediction == mnist.digits[i].value) {
				correct_predictions++;
			}

			propagateBackward(&net, target_output);
			count_samples++;

			if (count_samples == batch_size) {
				updateNetworkBatchParameters(&net, batch_size, learning_rate);
				count_samples = 0;
			}
		}

		//printf("Epoche [%d/%d] --> Genauigkeit auf Testdaten: %.2f%%\n", epoch, epochs, evaluateTestSet(&net, &mnist));

		float average_loss = epoch_loss_sum / (float)(mnist.total_digits);
		float train_accuracy = ((float)correct_predictions / mnist.total_digits) * 100.0f;
		printf("        Epoch [%2d/%2d] | Correct Predictions: %d | Average Loss: %f | Accuracy: %.2f%%\n", epoch, epochs, correct_predictions, average_loss, train_accuracy);

		shuffleDataset(&mnist);
	}
	printf("    }\n");
	printf("}\n");

	freeDataset(&mnist);
	readDataset(&mnist, "mnist_test.csv");

	printf("Epochs %d | Total Data Samples: %d | Genauigkeit auf Testdaten: %.2f%%\n", epochs, mnist.total_digits, evaluateTestSet(&net, &mnist));

	int user_input = 0;
	while (1) {
		printf("Enter a number between 0 - %d: ", mnist.total_digits - 1);
		if (scanf("%d", &user_input) == 0) {
			break;
		}

		int result = identifyDigit(&net, &mnist.digits[user_input]);
		printDigit(&mnist.digits[user_input], 1);
		printf("Neuronal Network result: %d\n", result);
	}

	freeDataset(&mnist);
	freeNeuronalNetwork(&net);
	return 0;
}
