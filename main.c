#include "headers/main.h"

int displaySaveNetworkDialog(NeuronalNetwork *nt);
int saveNeuronalNetwork(NeuronalNetwork *nt, const char filename[]);
int loadNeuronalNetwork(NeuronalNetwork *nt, const char filename[]);
int askYesOrNo(const char text[]);
void testNeuronalNetwork(NeuronalNetwork *nt, Dataset *dt, const char data_filename[]);

int main(int argv, char *argc[]) {
	Dataset mnist = { 0 };
	NeuronalNetwork net = { 0 };

	if (askYesOrNo("Load an existing network? [y/n]") == 0) {
		printf("Starting a new Neuronal Network...\n");
		readDataset(&mnist, "mnist_train.csv");
		if (!mnist.digits) {
			fprintf(stderr, "(mnist.digits == 0). No data have been loaded!");
			return -1;
		}

		createNeuronalNetwork(&net, 4);
		initNeuron(&net.neurons[inputNeuron], INPUT_LAYER, 0, mnist.pixels_pro_digit);
		initNeuron(&net.neurons[firstNeuron], FIRST_HIDDEN_LAYER, net.neurons[inputNeuron].num_of_perceptrons, 40);
		initNeuron(&net.neurons[secondNeuron], SECOND_HIDDEN_LAYER, net.neurons[firstNeuron].num_of_perceptrons, 40);
		initNeuron(&net.neurons[outputNeuron], OUTPUT_LAYER, net.neurons[secondNeuron].num_of_perceptrons, 10);
		net.epochs = 30;
		net.batch_size = 16;
		net.learning_rate = 0.002f;

		printNeuronalNetwork(&net);
		printf("Learning results {\n");

		for (int epoch = 0; epoch < net.epochs; epoch++) {
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

				int prediction = getNetworkPrediction(&net);
				if (prediction == mnist.digits[i].value) {
					correct_predictions++;
				}

				propagateBackward(&net, target_output);
				count_samples++;

				if (count_samples == net.batch_size) {
					updateNetworkBatchParameters(&net);
					count_samples = 0;
				}
			}

			float average_loss = epoch_loss_sum / (float)(mnist.total_digits);
			float train_accuracy = ((float)correct_predictions / mnist.total_digits) * 100.0f;
			printf("    Epoch [%2d/%2d] | Correct Predictions: %d | Average Loss: %f | Accuracy: %.2f%%\n", epoch, net.epochs, correct_predictions, average_loss, train_accuracy);

			shuffleDataset(&mnist);
		}
		printf("}\n");

		printf("Epochs %d | Total Data Samples: %d | Genauigkeit auf Testdaten: %.2f%%\n", net.epochs, mnist.total_digits, evaluateTestSet(&net, &mnist));
	} else {
		printf("Enter file (only name, without file extension and without spaces) from which the network will be loaded: ");
		char filename[128];
		if (scanf(" %s", &filename) != 0) {
			loadNeuronalNetwork(&net, filename);
		}
	}

	testNeuronalNetwork(&net, &mnist, "mnist_test.csv");

	displaySaveNetworkDialog(&net);

	freeDataset(&mnist);
	freeNeuronalNetwork(&net);
	return 0;
}
int displaySaveNetworkDialog(NeuronalNetwork *nt) {
	char input;
	while (1) {
		system("cls");
		printNeuronalNetwork(nt);
		printf("Do you want to save this Neuronal Network? [y/n] ");
		input = getc(stdin);
		while (getc(stdin) != '\n');

		if (input == 'y' || input == 'Y') {
			printf("Please enter a file name (without spaces and without file extension part) for your network: ");
			char filename[128];
			if (scanf_s(" %s", &filename, 128) != 0) {
				printf("Saving network in %s%s\n", PROJECT_MODELS_DIR, filename);
				while (getc(stdin) != '\n');
				return saveNeuronalNetwork(nt, filename);
			}
		} else if (input == 'n' || input == 'N') {
			printf("Exiting without save...\n");
			return 1;
		}
	}
	return 0;
}
int saveNeuronalNetwork(NeuronalNetwork *nt, const char filename[]) {
	const size_t path_length = strlen(filename) + strlen(PROJECT_MODELS_DIR) + 4 + 1; // +3 for the file extension and +1 for null terminating character.
	char *path = malloc(path_length);
	if (!path) {
		printf("Could not allocate memory for file: %s\n", filename);
		return 0;
	}
	sprintf_s(path, path_length, "%s%s.cnn", PROJECT_MODELS_DIR, filename);

	// Check if file already exists
	FILE *fp = fopen(path, "rb");
	if (fp) {
		if (askYesOrNo("File already exists. Overwrite? [y/n]") == 0) {
			fclose(fp);
			free(path);
			return 0;
		}
	}

	fp = fopen(path, "wb");
	fwrite(&nt->batch_size, 4, 1, fp);
	fwrite(&nt->epochs, 4, 1, fp);
	fwrite(&nt->learning_rate, 4, 1, fp);
	fwrite(&nt->total_neurons, 4, 1, fp);
	for (int active_neuron = 0; active_neuron < nt->total_neurons; active_neuron++) {
		fwrite(&nt->neurons[active_neuron].num_of_perceptrons, 4, 1, fp);
		fwrite(&nt->neurons[active_neuron].type, 4, 1, fp);
		fwrite(&nt->neurons[active_neuron].weights_dim_x, 4, 1, fp);
		fwrite(&nt->neurons[active_neuron].weights_dim_y, 4, 1, fp);
		for (int i = 0; i < nt->neurons[active_neuron].num_of_perceptrons; i++) {
			fwrite(&nt->neurons[active_neuron].perceptrons[i].bias, 4, 1, fp);
			fwrite(&nt->neurons[active_neuron].perceptrons[i].bias_acc, 4, 1, fp);
			fwrite(&nt->neurons[active_neuron].perceptrons[i].post_activation, 4, 1, fp);
			fwrite(&nt->neurons[active_neuron].perceptrons[i].pre_activation, 4, 1, fp);
		}
		if (nt->neurons[active_neuron].type != INPUT_LAYER) {
			for (int x = 0; x < nt->neurons[active_neuron].weights_dim_x; x++) {
				fwrite(nt->neurons[active_neuron].weights[x], (nt->neurons[active_neuron].weights_dim_y * sizeof(float)), 1, fp);
				fwrite(nt->neurons[active_neuron].weights_acc[x], (nt->neurons[active_neuron].weights_dim_y * sizeof(float)), 1, fp);
			}
		}
	}
	printf("Neuronal network have been writen!\n");
	fclose(fp);
	free(path);
	return 1;
}
int loadNeuronalNetwork(NeuronalNetwork *nt, const char filename[]) {
	const size_t path_length = strlen(filename) + strlen(PROJECT_MODELS_DIR) + 4 + 1; // +3 for the file extension and +1 for null terminating character.
	char *path = malloc(path_length);
	if (!path) {
		printf("Could not allocate memory for file: %s\n", filename);
		return 0;
	}
	sprintf_s(path, path_length, "%s%s.cnn", PROJECT_MODELS_DIR, filename);

	FILE *fp = fopen(path, "rb");
	if (!fp) {
		printf("Could not open file: %s\n", path);
		free(path);
		exit(1);
	}

	fread(&nt->batch_size, 4, 1, fp);
	fread(&nt->epochs, 4, 1, fp);
	fread(&nt->learning_rate, 4, 1, fp);
	fread(&nt->total_neurons, 4, 1, fp);
	nt->neurons = calloc(nt->total_neurons, sizeof(Neuron));
	if (!nt->neurons) {
		printf("Could not allocate memory for neurons. loadNeuronalNetwork()\n");
		fclose(fp);
		free(path);
		exit(1);
	}
	for (int active_neuron = 0; active_neuron < nt->total_neurons; active_neuron++) {
		fread(&nt->neurons[active_neuron].num_of_perceptrons, 4, 1, fp);
		fread(&nt->neurons[active_neuron].type, 4, 1, fp);
		fread(&nt->neurons[active_neuron].weights_dim_x, 4, 1, fp);
		fread(&nt->neurons[active_neuron].weights_dim_y, 4, 1, fp);

		nt->neurons[active_neuron].perceptrons = calloc(nt->neurons[active_neuron].num_of_perceptrons, sizeof(Perceptron));
		if (!nt->neurons[active_neuron].perceptrons) {
			printf("Could not allocate memory for perceptrons. loadNeuronalNetwork()\n");
			fclose(fp);
			free(path);
			exit(1);
		}
		for (int i = 0; i < nt->neurons[active_neuron].num_of_perceptrons; i++) {
			fread(&nt->neurons[active_neuron].perceptrons[i].bias, 4, 1, fp);
			fread(&nt->neurons[active_neuron].perceptrons[i].bias_acc, 4, 1, fp);
			fread(&nt->neurons[active_neuron].perceptrons[i].post_activation, 4, 1, fp);
			fread(&nt->neurons[active_neuron].perceptrons[i].pre_activation, 4, 1, fp);
		}

		if (nt->neurons[active_neuron].type != INPUT_LAYER) {
			nt->neurons[active_neuron].weights = calloc(nt->neurons[active_neuron].weights_dim_x, sizeof(double));
			nt->neurons[active_neuron].weights_acc = calloc(nt->neurons[active_neuron].weights_dim_x, sizeof(double));
			if (!nt->neurons[active_neuron].weights || !nt->neurons[active_neuron].weights_acc) {
				printf("Could not allocate memory for weights. loadNeuronalNetwork()\n");
				fclose(fp);
				free(path);
				exit(1);
			}
			for (int x = 0; x < nt->neurons[active_neuron].weights_dim_x; x++) {
				nt->neurons[active_neuron].weights[x] = calloc(nt->neurons[active_neuron].weights_dim_y, sizeof(float));
				nt->neurons[active_neuron].weights_acc[x] = calloc(nt->neurons[active_neuron].weights_dim_y, sizeof(float));
				if (!nt->neurons[active_neuron].weights[x] || !nt->neurons[active_neuron].weights_acc[x]) {
					printf("Could not allocate memory for weights[%d]. loadNeuronalNetwork()\n", x);
					fclose(fp);
					free(path);
					exit(1);
				}
				fread(nt->neurons[active_neuron].weights[x], (nt->neurons[active_neuron].weights_dim_y * sizeof(float)), 1, fp);
				fread(nt->neurons[active_neuron].weights_acc[x], (nt->neurons[active_neuron].weights_dim_y * sizeof(float)), 1, fp);
			}
		}
	}
	fclose(fp);
	free(path);
	return 0;
}
int askYesOrNo(const char text[]) {
	while (1) {
		printf("%s ", text);
		char input = getc(stdin);
		while (getc(stdin) != '\n');

		if (input == 'y' || input == 'Y') {
			return 1;
		} else if (input == 'n' || input == 'N') {
			return 0;
		}
	}
}
void testNeuronalNetwork(NeuronalNetwork *nt, Dataset *dt, const char data_filename[]) {
	readDataset(dt, data_filename);

	int user_input = 0;
	while (1) {
		printf("Enter a number between 0 - %d: ", dt->total_digits - 1);
		if (scanf("%d", &user_input) == 0) {
			break;
		}

		int result = identifyDigit(nt, &dt->digits[user_input]);
		printDigit(&dt->digits[user_input], 1);
		printf("Neuronal Network result: %d\n", result);
	}
}
