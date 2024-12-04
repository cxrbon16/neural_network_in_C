//
// Created by ayganyavuz on 15.11.2024.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 785 // (28 * 28) + 1
#define NUM_CLASSES 10

FILE* timeVsLoss;
FILE* weight1ToN;
FILE* tenTanhAccuracy;

typedef struct dataPoint{
  double* X;
  int dim;
  int* Y; 
} dataPoint;

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
    weights[i] = randFrom(-0.5, 0.5);
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

double computeAccuracy(dataPoint** testSet, int testSize, double** weights){
  double* Yhat;
  int* targetY;
  double maxOutput;
  int outputIndex;
  int correct;
  for(int i = 0; i < testSize; i++){
    Yhat = forwardProp(testSet[i], weights);
    targetY = testSet[i]->Y;
    maxOutput = Yhat[0];
    outputIndex = 0;
    for(int j = 1; j < NUM_CLASSES; j++){
      if (Yhat[j] > maxOutput){
        outputIndex = j;
        maxOutput = Yhat[j];
      }
    }
    for(int j = 0; j < NUM_CLASSES; j++){
      if(targetY[j] == 1 && j == outputIndex){
        correct++;
      }
    }
  }
  return (double) correct / (double) testSize;
}

double computeLoss(dataPoint** set, int size, double** weights){
  double totalLoss = 0.0;
  double* yHat;
  int* targetY;
  for(int i = 0; i < size; i++){
    yHat = forwardProp(set[i], weights);
    targetY = set[i]->Y;
    for(int j = 0; j < NUM_CLASSES; j++)
      totalLoss += pow(yHat[j] - targetY[j], 2);
  }
  return totalLoss / size;
}

void gradientDescent(dataPoint** dataPoints, dataPoint** dataPointsTest, int trainSize, int testSize,  double** weights, double alpha, int epoch){

  // Gradientleri bulmak için:
  // Loss[0] = 1/n * Sum((Yhat[0] - Y[0]) ^ 2)
  // Dloss[0]/DW[0] = 2/n * Yhat[0] - Y[0] * TanhDiff(WX) * X
  // Dloss[0]/DW[0] uzunluğu INPUT_SIZE'a yani Weightlerin boyutuna eşittir ki bu aslında W'nun loss'a göre olan kısmı türevi/gradyanıdır.

  double** gradients = malloc(NUM_CLASSES * sizeof(double*));
  for(int i = 0; i < NUM_CLASSES; i++){
    gradients[i] = malloc(INPUT_SIZE * sizeof(double));
  }
  double preActivationWX;
  double* Y = (double*) malloc(NUM_CLASSES * sizeof(double));
  double tanhDiffVal;
  double Loss = 0;


  clock_t start_time = clock();
  clock_t curr_time;
  double currTime;

  for (int iter_num = 0; iter_num < epoch; iter_num++){

    for(int k = 0; k < NUM_CLASSES; k++)
      for(int i = 0; i < INPUT_SIZE; i++)
        gradients[k][i] = 0.0;

    for (int i = 0; i < trainSize; i++) {

        for(int j = 0; j < NUM_CLASSES; j++) {
          preActivationWX = linearForward(weights[j], dataPoints[i]->X, INPUT_SIZE);
          Y[j] = tanhForwardSingle(preActivationWX);
          tanhDiffVal = tanhDiffSingle(preActivationWX);
          for(int l = 0; l < INPUT_SIZE; l++){
            gradients[j][l] += dataPoints[i]->X[l] *  tanhDiffVal * (Y[j] - (double) dataPoints[i]->Y[j]) * (double) (2.0/trainSize);
          }
        }
    }

    
    curr_time = clock() - start_time;
    currTime = ((double) curr_time) / CLOCKS_PER_SEC;
    fprintf(timeVsLoss, "%f %f %f %f\n", currTime, (double) iter_num, computeLoss(dataPoints, trainSize, weights), 
                                        computeLoss(dataPointsTest, testSize, weights));

    if (iter_num % 1 == 0){
      printf("EPOCH: %d\n", iter_num);
      printf("TRAINSET LOSS: %f\n", computeLoss(dataPoints, trainSize, weights));
      printf("TESTSET LOSS: %f\n", computeLoss(dataPointsTest, testSize, weights));
      printf("ACCURACY ON TESTSET: %f\n\n\n", computeAccuracy(dataPointsTest, testSize, weights));
    }

    for (int i = 0; i < NUM_CLASSES; i++){
      for(int j = 0; j < INPUT_SIZE; j++){
        weights[i][j] = weights[i][j] - alpha * gradients[i][j] ;
        // printf("%f\n", gradients[i][j]);
      }
    }
  
  }
  for (int i = 0; i < NUM_CLASSES; i++) {
    free(gradients[i]);
  }
  free(gradients);
  free(Y);
}

void stochasticGradientDescent(dataPoint** dataPoints, dataPoint** dataPointsTest, int trainSize, int testSize,  double** weights, double alpha, int epoch){
  double** gradients = malloc(NUM_CLASSES * sizeof(double*));
  for(int i = 0; i < NUM_CLASSES; i++){
    gradients[i] = malloc(INPUT_SIZE * sizeof(double));
  }
  double preActivationWX;
  double* Y = (double*) malloc(NUM_CLASSES * sizeof(double));
  double tanhDiffVal;
  double Loss = 0;
  double testLossShifted = 0;

  int randomIndex;
  double tmpLoss = 0;

  clock_t start_time = clock();
  clock_t curr_time;
  double currTime;

  for (int iter_num = 0; iter_num < epoch * trainSize; iter_num++){
    for(int k = 0; k < NUM_CLASSES; k++)
      for(int i = 0; i < INPUT_SIZE; i++)
        gradients[k][i] = 0.0;

    randomIndex = rand() % trainSize;;
    tmpLoss = 0.0;

    for(int j = 0; j < NUM_CLASSES; j++) {
          preActivationWX = linearForward(weights[j], dataPoints[randomIndex]->X, INPUT_SIZE);
          Y[j] = tanhForwardSingle(preActivationWX);
          tanhDiffVal = tanhDiffSingle(preActivationWX);
          for(int l = 0; l < INPUT_SIZE; l++){
            gradients[j][l] = dataPoints[randomIndex]->X[l] *  tanhDiffVal * (Y[j] - (double) dataPoints[randomIndex]->Y[j]) * 2.0;
            weights[j][l] = weights[j][l] - alpha * gradients[j][l];
          }
          tmpLoss += (Y[j] - dataPoints[randomIndex]->Y[j]);

    }
    Loss += pow(tmpLoss, 2);

    //AĞIRLIKLARIN KAYDEDİLMESİ.
    if (iter_num % trainSize == 0) {  
      fprintf(weight1ToN, "EPOCH %d\n", iter_num / trainSize);
      for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
          fprintf(weight1ToN, "%f ", weights[i][j]);
        }
      fprintf(weight1ToN, "\n");
      }
    } 
    if(iter_num % trainSize == 0){
      fprintf(tenTanhAccuracy, "%f\n", computeAccuracy(dataPointsTest, testSize, weights));
    }
    // ZAMAN, EPOCH, TRAINLOSS, TESTLOSSların kaydedilmesi.
    if(iter_num % trainSize == 0){
      curr_time = clock() - start_time;
      currTime = ((double) curr_time) / CLOCKS_PER_SEC;
      fprintf(timeVsLoss, "%f %f %f %f\n", currTime, (double) iter_num / trainSize, computeLoss(dataPoints, trainSize, weights),
                                        computeLoss(dataPointsTest, testSize, weights));
    }
    // OPTIMIZASYON SÜRECİNİN TAKİBİ İÇİN EKRANA LOGLAMA İŞLERİ. 
    if (iter_num % trainSize == 0){
      printf("EPOCH: %d\n", iter_num / trainSize);
      printf("TRAINSET LOSS: %f\n", computeLoss(dataPoints, trainSize, weights));
      printf("TESTSET LOSS: %f\n", computeLoss(dataPointsTest, testSize, weights));
      printf("ACCURACY ON TESTSET: %f\n\n\n", computeAccuracy(dataPointsTest, testSize, weights));
    }
    for (int i = 0; i < NUM_CLASSES; i++){
      for(int j = 0; j < INPUT_SIZE; j++){
        // printf("%f\n", gradients[i][j]);
      }
    }
  }
  for (int i = 0; i < NUM_CLASSES; i++) {
    free(gradients[i]);
  }
  free(gradients);
  free(Y);
}

void adamOptimizer(dataPoint** dataPoints, dataPoint** dataPointsTest, int trainSize, int testSize, double** weights, double alpha, int epoch) {

  double** m = malloc(NUM_CLASSES * sizeof(double*));  // First moment
  double** v = malloc(NUM_CLASSES * sizeof(double*));  // Second moment
  for (int i = 0; i < NUM_CLASSES; i++) {
    m[i] = calloc(INPUT_SIZE, sizeof(double));
    v[i] = calloc(INPUT_SIZE, sizeof(double));
  }
  double beta1 = 0.9;
  double beta2 = 0.99;
  double epsilon = 1e-8;
  int t = 0;

  double preActivationWX;
  double* Y = malloc(NUM_CLASSES * sizeof(double));
  double tanhDiffVal;
  double Loss = 0;
  double testLossShifted = 0;

  int randomIndex;
  double tmpLoss = 0;

  time_t start_time = clock();
  time_t curr_time;
  double curr_time_d;

  for (int iter_num = 0; iter_num < epoch * trainSize; iter_num++) {
    randomIndex = rand() % trainSize;
    tmpLoss = 0.0;

    for (int j = 0; j < NUM_CLASSES; j++) {
      preActivationWX = linearForward(weights[j], dataPoints[randomIndex]->X, INPUT_SIZE);
      Y[j] = tanhForwardSingle(preActivationWX);
      tanhDiffVal = tanhDiffSingle(preActivationWX);

      for (int l = 0; l < INPUT_SIZE; l++) {
        double gradient = dataPoints[randomIndex]->X[l] * tanhDiffVal * (Y[j] - dataPoints[randomIndex]->Y[j]) * 2.0;

        m[j][l] = beta1 * m[j][l] + (1 - beta1) * gradient;
        v[j][l] = beta2 * v[j][l] + (1 - beta2) * gradient * gradient;

        double m_hat = m[j][l] / (1 - pow(beta1, t + 1));
        double v_hat = v[j][l] / (1 - pow(beta2, t + 1));

        weights[j][l] = weights[j][l] - alpha * m_hat / (sqrt(v_hat) + epsilon);
      }
      tmpLoss += pow(Y[j] - dataPoints[randomIndex]->Y[j], 2);
    }

    Loss += tmpLoss;
    if (iter_num % trainSize == trainSize / 2 || iter_num == 0) {
      testLossShifted = computeLoss(dataPointsTest, testSize, weights);
    }
  if(iter_num % trainSize == 0){
      curr_time = clock() - start_time;
      curr_time_d = ((double) curr_time) / CLOCKS_PER_SEC;
      fprintf(timeVsLoss, "%f %f %f %f\n", curr_time_d, (double) iter_num / trainSize, computeLoss(dataPoints, trainSize, weights),
                                          computeLoss(dataPointsTest, testSize, weights));
    }
    if (iter_num % trainSize == 0) {

      printf("EPOCH: %d\n", iter_num / trainSize);
      printf("TRAINSET LOSS: %f\n", computeLoss(dataPoints, trainSize, weights));
      printf("TESTSET LOSS: %f\n", testLossShifted);
      printf("ACCURACY ON TESTSET: %f\n\n\n", computeAccuracy(dataPointsTest, testSize, weights));
    }
    t++;
  }

  // Free allocated memory
  for (int i = 0; i < NUM_CLASSES; i++) {
    free(m[i]);
    free(v[i]);
  }
  free(m);
  free(v);
  free(Y);
}


int main() {
  // Example usage

  int trainSize = 59000;
  int testSize = 9900;

  char* filePath = "data/train10Classes.txt";
  char* filePathTest = "data/test10Classes.txt";
  
  dataPoint** dataPoints = readInput(trainSize, filePath);
  dataPoint** dataPointsTest = readInput(testSize, filePathTest);

  weight1ToN = fopen("weight1toN.txt", "w");
  timeVsLoss = fopen("timeVsLoss.txt", "w");
  tenTanhAccuracy = fopen("10tanhAccuracy.txt", "w");

  double** weights = initializeWeightsMatrix();

  // gradientDescent(dataPoints, dataPointsTest, trainSize, testSize, weights, 0.05, 20);
  stochasticGradientDescent(dataPoints, dataPointsTest, trainSize, testSize, weights, 0.05, 10);
  // adamOptimizer(dataPoints, dataPointsTest, trainSize, testSize, weights, 0.05, 20);


  fclose(timeVsLoss);
  fclose(weight1ToN);

  return 0;
}