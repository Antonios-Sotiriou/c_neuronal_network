#ifndef DATASET_H
#define DATASET_H 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Project specific headers
#ifndef CMAKE_VARIABLES_H
    #include "cmake_variables.h"
#endif

typedef struct {
	int value;
	int pixels[784];
} Digit;

typedef struct {
	Digit *digits;
	int total_digits;
	int pixels_pro_digit;
} Dataset;

char *createRelativePath(const char filename[]);
int countFileLines(const char file_path[]);
void readDataset(Dataset *dt, const char filename[]);
void parseLine(Digit *dg, char line[]);
void shuffleDataset(Dataset *dt);
void printDigit(Digit *dg, const int show_bg);
void printDataset(Dataset *dt, const int show_bg);
void freeDataset(Dataset *dt);

#endif // !DATASET_H