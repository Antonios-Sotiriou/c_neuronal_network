#include "headers/dataset.h"

char *createRelativePath(const char filename[]) {
	size_t path_len = strlen(PROJECT_DATA_DIR) + strlen(filename) + 1;

	char *path = malloc(path_len);
	if (!path) {
		return NULL;
	}
	snprintf(path, path_len, "%s%s", PROJECT_DATA_DIR, filename);

	return path;
}
int countFileLines(const char file_path[]) {
	int lines = 0;
	FILE *fp = fopen(file_path, "r");
	if (fp) {
		char line[4096] = { 0 };
		while (fscanf_s(fp, "%[^\n] ", line, 4096) == 1) {
			lines++;
		}
		fclose(fp);
	}

	return lines;
}
void readDataset(Dataset *dt, const char filename[]) {
	char *file_path = createRelativePath(filename);

	int total_entries = countFileLines(file_path);
	printf("Lines in file: %d\n", total_entries);

	dt->digits = malloc(sizeof(Digit) * total_entries);
	if (!dt->digits) {
		fprintf(stderr, "Could not allocate memory for digits");
		return;
	}
	dt->total_digits = total_entries;
	dt->pixels_pro_digit = 784;

	FILE *fp = fopen(file_path, "r");
	if (fp) {
		char line[4096] = { 0 };
		int index = 0;
		while (fscanf_s(fp, "%[^\n] ", line, 4096) == 1) {
			parseLine(&dt->digits[index], line);
			index++;
		}
		fclose(fp);
	}

	free(file_path);
}
void parseLine(Digit *dg, char line[]) {

	char *context = NULL;
	char *token = strtok_s(line, ",", &context);
	if (token != NULL) {
		dg->value = atoi(token);
		token = strtok_s(NULL, ",", &context);
	}

	int index = 0;
	while (token != NULL) {
		dg->pixels[index] = atoi(token);
		index++;
		token = strtok_s(NULL, ",", &context);
	}
}
// Fisher-Yates algorithm.
void shuffleDataset(Dataset *dt) {
	for (int i = dt->total_digits - 1; i > 0; i--) {

		int j = rand() % (i + 1);

		Digit temp = dt->digits[i];
		dt->digits[i] = dt->digits[j];
		dt->digits[j] = temp;
	}
}
void printDigit(Digit *dg, const int show_bg) {
	int cols = 0;
	for (int i = 0; i < 784; i++) {
		if (show_bg) {
			printf("%03d", dg->pixels[i]);
		} else {
			if (dg->pixels[i] == 0) {
				printf("   ");
			} else {
				printf("%03d", dg->pixels[i]);
			}
		}

		cols++;
		if (cols == 28) {
			printf("\n");
			cols = 0;
		}
	}
	printf("Digit value: %d\n", dg->value);
}
void printDataset(Dataset *dt, const int show_bg) {
	for (int i = 0; i < dt->total_digits; i++) {
		printDigit(&dt->digits[i], show_bg);
	}
}
void freeDataset(Dataset *dt) {
	free(dt->digits);
}