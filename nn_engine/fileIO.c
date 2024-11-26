#include "nn_engine.h"

dataPoint* readInputHelperFunc(char* buffer, int numClasses, int numParameters) {
    char* newBuffer = strdup(buffer); // Duplicate buffer to safely tokenize
    if (!newBuffer) {
        printf("Memory allocation failed for newBuffer.\n");
        return NULL;
    }

    // Parse the label
    char* labelToken = strtok(newBuffer, " ");
    int label = labelToken[0] - '0'; 

    // Parse the input data
    double* dataX = (double*)malloc(numParameters * sizeof(double));
    if (!dataX) {
        printf("Memory allocation failed for dataX.\n");
        free(newBuffer);
        return NULL;
    }

    int i = 0;
    char* token;
    while ((token = strtok(NULL, " ")) && i < numParameters) {
        dataX[i++] = atof(token);
    }

    if (i != numParameters) {
        printf("Error: Expected %d data points, but got %d.\n", numParameters, i);
        free(dataX);
        free(newBuffer);
        return NULL;
    }

    // Create tensor for dataX
    int inputShape[2] = {1, numParameters};
    Tensor* inputTensor = createTensor(dataX, inputShape, 2, numParameters);

    // Create one-hot encoded tensor for the label
    double* onehotY = (double*)malloc(numClasses * sizeof(double));
    if (!onehotY) {
        printf("Memory allocation failed for onehotY.\n");
        freeTensor(inputTensor);
        free(newBuffer);
        return NULL;
    }

    for (int j = 0; j < numClasses; j++) {
        onehotY[j] = (label == j) ? 1.0 : -1.0;
    }
    int labelShape[2] = {1, numClasses};
    Tensor* labelTensor = createTensor(onehotY, labelShape, 2, numClasses);

    // Combine X and Y into a dataPoint struct
    dataPoint* result = (dataPoint*)malloc(sizeof(dataPoint));
    if (!result) {
        printf("Memory allocation failed for dataPoint struct.\n");
        freeTensor(inputTensor);
        freeTensor(labelTensor);
        free(newBuffer);
        return NULL;
    }

    result->X = inputTensor;
    result->Y = labelTensor;

    free(newBuffer);
    return result;
}

dataPoint** readInput(int numPoints, int numParameters, int numClasses, char* filePath) {
    if (!filePath) {
        printf("Error: File path is NULL.\n");
        return NULL;
    }

    dataPoint** resultArray = (dataPoint**)malloc(numPoints * sizeof(dataPoint*));
    if (!resultArray) {
        printf("Memory allocation failed for resultArray.\n");
        return NULL;
    }

    FILE* fptr = fopen(filePath, "r");
    if (!fptr) {
        printf("Error opening file: %s\n", filePath);
        free(resultArray);
        return NULL;
    }

    int bufferSize = (numParameters * 20); // Allocate enough space for a line of input
    char buffer[bufferSize];

    int i = 0;
    while (fgets(buffer, bufferSize, fptr) && i < numPoints) {
        dataPoint* dataPointTensor = readInputHelperFunc(buffer, numClasses, numParameters);
        if (!dataPointTensor) {
            printf("Error reading data point at index %d.\n", i);
            continue;
        }
        resultArray[i++] = dataPointTensor;
    }

    fclose(fptr);
    return resultArray;
}
