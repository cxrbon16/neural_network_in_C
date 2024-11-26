/**
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : nn_engine
 * @created     : Pazartesi Kas 25, 2024 00:03:26 +03
 */

#include "nn_engine.h"
#include "../tensor_engine/tensor_engine.h"


Layer* initializeLayer(int numNodes, int inputDim, double (*randomFunc)(double, double), double (*activationFunction)(double), double (*activationDerivativeFunction)(double)){
  Layer* resultLayer = malloc(sizeof(Layer));
  resultLayer->numNodes = numNodes;
  resultLayer->inputDim = inputDim;

  int* shape = malloc(2 * sizeof(int));
  // CAREFUL
  shape[0] = inputDim;shape[1] = numNodes; 

  Tensor* tensor = randomTensor(shape, 2, shape[0] * shape[1], NULL);
  free(shape);
  resultLayer->layerTensor = tensor;
  resultLayer->activationFunction = activationFunction;
  resultLayer->activationDerivativeFunction = activationDerivativeFunction;

  return resultLayer;
}



Tensor* forwardLayer(Layer *layer, Tensor *inputTensor){
  if(layer->inputDim != inputTensor->shape[1]){
    printf("\n something went wrong with forwarding layers.");
    return NULL;
  }
  Tensor* resultTensor;
  resultTensor = tensorTensorMUL(inputTensor, layer->layerTensor);

  applyFuncToTensor(resultTensor, layer->activationFunction);
  return resultTensor;
}


Tensor* forwardMLP(MLP* mlp, Tensor* inputTensor){
  Tensor* currentTensor = inputTensor;
  for (int i = 0; i < mlp->numLayers; i++) {
    Tensor* nextTensor = forwardLayer(mlp->layers[i], currentTensor);

    free(mlp->cacheActivations[i]);
    mlp->cacheActivations[i] = nextTensor;
    /*
    if (currentTensor != inputTensor) {
      free(mlp->cacheActivations[i-1]);
      mlp->cacheActivations[i-1] = nextTensor;
    }
  */
    currentTensor = nextTensor;
  }
  return currentTensor; 
}

Tensor* backwardLayer(Layer* layer, Tensor* inputTensor, Tensor* outputGrad){
  // preActivation = tensorTensorMUL(inputTensor, layer->layerTensor) olsun.
  // F = activationFunction(preActivation) = outputTensor 
  // DLoss/DF'in bize verildiğini varsayarsak, Tensor* outputGrad
  // DLoss/DlayerTensor'ü arıyoruz.
  // DLoss/DlayerTensor = DLoss/DF * DF/DpreActivation * DpreActivation/DlayerTensor
  Tensor* preActivation = tensorTensorMUL(inputTensor, layer->layerTensor);

  applyFuncToTensor(preActivation, layer->activationDerivativeFunction);

  Tensor* delta = tensorTensorElementWiseMUL(outputGrad, preActivation);

  
  transposeTensor(inputTensor);
  Tensor* gradientTensor = tensorTensorMUL(inputTensor, delta);
  transposeTensor(inputTensor);

  if(layer->gradientTensor != NULL){
    layer->gradientTensor = addTensors(layer->gradientTensor, gradientTensor);
  }else{
    layer->gradientTensor = gradientTensor;
  }
  transposeTensor(layer->layerTensor);
  Tensor* tensorInputGradient  = tensorTensorMUL(delta, layer->layerTensor);
  transposeTensor(layer->layerTensor);
  freeTensor(delta);
  return tensorInputGradient;
}



void computeGradients(MLP* mlp, Tensor* inputTensor, Tensor* targetTensor){

  Tensor* output = forwardMLP(mlp, inputTensor);

  Tensor* lossGradient = computeLossGradient(output, targetTensor, mlp->costDerivativeFunction);
  Tensor* tmpGrad = lossGradient;
  for(int i = mlp->numLayers - 1; i >= 0; i--){
    Tensor* prevInput = (i == 0) ? inputTensor : mlp->cacheActivations[i-1];
    tmpGrad = backwardLayer(mlp->layers[i], prevInput, tmpGrad);
  }
}

Tensor* computeLossGradient(Tensor* yhat, Tensor* y, double (*costDerivativeFunction)(double, double)){
  Tensor* gradient = copyTensor(yhat);
  for(int i = 0; i < yhat->numElements; i++){
    gradient->elements[i] = costDerivativeFunction(yhat->elements[i], y->elements[i]);
  }
  return gradient;
}

void zeroGradients(Layer* layer){
  for(int i = 0; i < layer->gradientTensor->numElements; i++){
    layer->gradientTensor->elements[i] = 0.0;
  }
}
/*
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
*/