//
// Created by ayganyavuz on 15.11.2024.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 785 // (28 * 28) + 1
#define NUM_CLASSES 4

typedef struct dataPoint{
  double* X;
  int dim;
  int* Y; 
} dataPoint;

typedef struct Layer{
  double* input;
  double* output;
  int numNodes;
  char* activationType;
} Layer;

double* tanhForward(double* X, int inputSize){
  double* outputArray = (double*)malloc(inputSize * sizeof(double));
  for (int i = 0; i < inputSize; i++) {
    outputArray[i] = tanh(X[i]);
  }
  return outputArray;
}

double* tanhDiff(double* X, int XSize){
  double* outputArray = (double*)malloc(XSize * sizeof(double));
  for (int i = 0; i < XSize; i++) {
    outputArray[i] = 1 - tanh(X[i]) * tanh(X[i]);
  }
  return outputArray;
}

double linearForward(double *X, double *W, int XSize){
  double result = 0;
  for (int i = 0; i < XSize; i++) {
    result += X[i] * W[i];
  }
  return result;
}
double* linearDW(double *X, double *W, int XSize){
  // (dF(W, X) / dW)
  // Differention of linear function of X and W with respect to W.

  double* outputArray = (double*)malloc(XSize * sizeof(double));
  for (int i = 0; i < XSize; i++) {
    outputArray[i] = X[i];
  }
  return outputArray;
}

dataPoint* readInputHelperFunc(char* buffer, int bufferSize) {

  dataPoint* result = (dataPoint*)malloc(sizeof(dataPoint));
  char* newBuffer = strdup(buffer); // Duplicate buffer to safely tokenize


  char* labelToken = strtok(newBuffer, " ");
  int label = labelToken[0] - '0'; 

  // Parse data points
  double* dataX = (double*)malloc(INPUT_SIZE * sizeof(double));

  int i = 0;
  char* token;
  while ((token = strtok(NULL, " ")) && i < INPUT_SIZE) {
    dataX[i++] = atof(token);
  }

  if (i != INPUT_SIZE) {
    printf("Error: Expected %d data points, but got %d.\n", INPUT_SIZE, i);
    free(dataX);
    free(newBuffer);
    free(result);
    return NULL;
  }
  
  int* onehotY = (int*)malloc(NUM_CLASSES * sizeof(int));
  for (int i = 0; i < NUM_CLASSES; i++) {
    onehotY[i] = (label == i) ? 1 : -1;
  }

  result->X = dataX;
  result->dim = INPUT_SIZE;
  result->Y = onehotY;

  free(newBuffer);
  return result;
}


dataPoint** readInput(int inputSize, char* filePath) {
  if (!filePath) {
    printf("Error: File path is NULL.\n");
    return NULL;
  }

  dataPoint** resultArray = (dataPoint**)malloc(inputSize * sizeof(dataPoint*));
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

  int bufferSize = (INPUT_SIZE * 20); // Her bir double için yaklaşık 20 char'lık yer ayırıyoruz.
  char buffer[bufferSize];

  int i = 0;
  while (fgets(buffer, bufferSize, fptr) && i < inputSize) {
    dataPoint* dataPoint = readInputHelperFunc(buffer, bufferSize);
    if (!dataPoint) {
      printf("Error reading data point at index %d.\n", i);
      continue;
    }
    resultArray[i++] = dataPoint;
  }

  fclose(fptr);
  return resultArray;
}

double randFrom(double min, double max){
  return min + (max - min) * (rand() / (RAND_MAX + 1.0));
}

double* initializeWeights(int size) {
  double* weights = (double*)malloc(size * sizeof(double));
  // Xavier initialization: range is [-1/sqrt(n), 1/sqrt(n)]
  srand(time(0));

  for (int i = 0; i < size; i++) {
    weights[i] = randFrom(-0.1, 0.1);
  }
  return weights;
}




double** initializeWeightsMatrix(){
  double** weights = malloc(NUM_CLASSES * sizeof(double*));
  for (int i = 0; i < NUM_CLASSES; i++) {
    weights[i] = initializeWeights(INPUT_SIZE);
  }
  return weights;
}

double tanhForwardSingle(double x) {
  return tanh(x);
}


double tanhDiffSingle(double x) {
  return 1 - tanh(x) * tanh(x);
}

double* forwardProp(dataPoint* dataPoint, double** weights){
  double* Y = malloc(NUM_CLASSES * sizeof(double));
  for(int i = 0; i < NUM_CLASSES; i++) {
    Y[i] = linearForward(dataPoint->X, weights[i], INPUT_SIZE);
    Y[i] = tanhForwardSingle(Y[i]);
  }
  return Y;
}


int main() {
  // Example usage

  int trainSize = 20000;
  char* filePath = "data/train.txt";

  dataPoint** dataPoints = readInput(trainSize, filePath);
  double** weights = initializeWeightsMatrix();

  // lets find the gradients.
  // Loss[0] is 1/n * Sum(Yhat[0] - Y[0])
  // Dloss[0]/DW[0] = 2/n * (Sum(Yhat[0] - Y[0]) * TanhDiff(WX) * X
  // the equation above is a vector whose size is N^2 + 1 or INPUT_SIZE

  double** gradients = malloc(NUM_CLASSES * sizeof(double*));
  for(int i = 0; i < NUM_CLASSES; i++){
    gradients[i] = malloc(INPUT_SIZE * sizeof(double));
  }
  double* sum = malloc(NUM_CLASSES * sizeof(double)); 
  double preActivationWX;
  double* Y = (double*) malloc(NUM_CLASSES * sizeof(double));
  double tanhDiffVal;
  int epoch = 100;
  double Loss = 0;
  for (int iter_num = 0; iter_num < epoch; iter_num++){

    for(int k = 0; k < NUM_CLASSES; k++){
      sum[k] = 0.0;
      for(int i = 0; i < INPUT_SIZE; i++)
        gradients[k][i] = 0.0;
    }
    Loss = 0;
    for (int i = 0; i < trainSize; i++) {

        for(int j = 0; j < NUM_CLASSES; j++) {
          preActivationWX = linearForward(weights[j], dataPoints[i]->X, INPUT_SIZE);
          Y[j] = tanhForwardSingle(preActivationWX);
          tanhDiffVal = tanhDiffSingle(preActivationWX);
          for(int l = 0; l < INPUT_SIZE; l++){
            gradients[j][l] += dataPoints[i]->X[l] *  tanhDiffVal * (Y[j] - dataPoints[i]->Y[j]) * (double) (2.0/trainSize);
          }
          Loss += pow((Y[j] - dataPoints[i]->Y[j]), 2);
        }
    }
    double alpha = 0.1;
    for (int i = 0; i < NUM_CLASSES; i++){
      for(int j = 0; j < INPUT_SIZE; j++){
        weights[i][j] = weights[i][j] - alpha * gradients[i][j] ;
        // printf("%f\n", gradients[i][j]);
      }
    }

    printf("%f\n", Loss/trainSize);

  }

  int testSize = 10;
  char* filePathTest = "data/test.txt";
  double *Yhat;
  dataPoint** dataPointsTest = readInput(testSize, filePathTest);

  for(int i = 0; i < testSize; i++){
    Yhat = forwardProp(dataPointsTest[i], weights);
    for(int j = 0; j < NUM_CLASSES; j++){
      printf("Yhat[%d]: %f Y[%d]: %d\n", j, Yhat[j], j,  dataPointsTest[i]->Y[j]);
      //printf("%i\n", dataPointsTest[i]->Y[j]);
    }
  }
  return 0;
}
