#include "headers/main.h"

int main(int argv, char *argc[]) {

	Dataset mnist = { 0 };

	readDataset(&mnist, "mnist_train.csv");
	if (!mnist.digits) {
		fprintf(stderr, "(mnist.digits == 0). No data have been loaded!");
		return -1;
	}

	Neuron inputNeuron = { 0 };
	Neuron firstNeuron = { 0 };
	Neuron secondNeuron = { 0 };
	Neuron outputNeuron = { 0 };
	initNeuron(&inputNeuron, INPUT_LAYER, 0, mnist.pixels_pro_digit);
	initNeuron(&firstNeuron, FIRST_HIDDEN_LAYER, inputNeuron.num_of_perceptrons, 128);
	initNeuron(&secondNeuron, SECOND_HIDDEN_LAYER, firstNeuron.num_of_perceptrons, 128);
	initNeuron(&outputNeuron, OUTPUT_LAYER, secondNeuron.num_of_perceptrons, 10);

	int epochs = 30;
	int batch_size = 20;
	float learning_rate = 0.01;

	printf("Neuronal Network {\n");
	printf("    Epochs              : %d\n", epochs);
	printf("    batch Size          : %d\n", batch_size);
	printf("    Learning Rate       : %f\n", learning_rate);
	printf("    Input Perceptrons   : %d\n", inputNeuron.num_of_perceptrons);
	printf("    Layer 1 Perceptrons : %d\n", firstNeuron.num_of_perceptrons);
	printf("    Layer 2 Perceptrons : %d\n", secondNeuron.num_of_perceptrons);
	printf("    Output Perceptrons  : %d\n", outputNeuron.num_of_perceptrons);
	printf("    Samples {\n");

	for (int epoch = 0; epoch < epochs; epoch++) {
		float epoch_loss_sum = 0.0f;
		int correct_predictions = 0;
		int count_samples = 0;

		for (int i = 0; i < mnist.total_digits; i++) {

			initInputNeuron(&inputNeuron, &mnist.digits[i]);
			propagateForward(&inputNeuron, &firstNeuron);
			propagateForward(&firstNeuron, &secondNeuron);
			propagateForward(&secondNeuron, &outputNeuron);

			int target_output[10] = { 0 };
			target_output[mnist.digits[i].value] = 1;
			float loss = lossEntropy(&outputNeuron, target_output);
			epoch_loss_sum += loss;

			//printf("Computed loss: %f\n", loss);
			int prediction = getNetworkPrediction(&outputNeuron);
			if (prediction == mnist.digits[i].value) {
				correct_predictions++;
			}

			propagateBackward(&outputNeuron, &secondNeuron, &firstNeuron, &inputNeuron, target_output);
			count_samples++;

			if (count_samples == batch_size) {
				updateBatchParameters(&outputNeuron, batch_size, learning_rate);
				updateBatchParameters(&secondNeuron, batch_size, learning_rate);
				updateBatchParameters(&firstNeuron, batch_size, learning_rate);
				count_samples = 0;
			}
		}

		//float test_accuracy = evaluateTestSet(&inputNeuron, &firstNeuron, &secondNeuron, &outputNeuron, &mnist);
		//printf("Epoche [%d/%d] --> Genauigkeit auf Testdaten: %.2f%%\n", epoch, epochs, test_accuracy);

		float average_loss = epoch_loss_sum / (float)(mnist.total_digits);
		float train_accuracy = ((float)correct_predictions / mnist.total_digits) * 100.0f;
		printf("        Epoch [%2d/%2d] | Correct Predictions: %d | Average Loss: %f | Accuracy: %.2f%%\n", epoch, epochs, correct_predictions, average_loss, train_accuracy);

		shuffleDataset(&mnist);
	}
	printf("    }\n");
	printf("}\n");

	float test_accuracy = evaluateTestSet(&inputNeuron, &firstNeuron, &secondNeuron, &outputNeuron, &mnist);
	printf("Epochs %d | Genauigkeit auf Testdaten: %.2f%%\n", epochs, test_accuracy);

	int user_input = 0;
	while (1) {
		printf("Enter a number between 0 - %d: ", mnist.total_digits - 1);
		if (scanf("%d", &user_input) == 0) {
			break;
		}

		int result = identifyDigit(&inputNeuron, &firstNeuron, &secondNeuron, &outputNeuron, &mnist.digits[user_input]);
		printDigit(&mnist.digits[user_input], 1);
		printf("Neuronal Network result: %d\n", result);
	}

	//printDigit(&mnist.digits[1], 0);
	//printDataset(&mnist);

	freeDataset(&mnist);
	freeNeuron(&inputNeuron);
	freeNeuron(&firstNeuron);
	freeNeuron(&outputNeuron);
	return 0;
}
