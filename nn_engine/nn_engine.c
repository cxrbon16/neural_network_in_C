/**
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : nn_engine
 * @created     : Pazartesi Kas 25, 2024 00:03:26 +03
 */

#include "nn_engine.h"
#include "../tensor_engine/tensor_engine.h"
#include <math.h>

void clipGradients(Tensor* gradientTensor, double maxNorm) {
    // Compute gradient norm (L2 norm)
    double gradientNorm = 0.0;
    for (int i = 0; i < gradientTensor->numElements; i++) {
        gradientNorm += gradientTensor->elements[i] * gradientTensor->elements[i];
    }
    gradientNorm = sqrt(gradientNorm);

    // If gradient norm exceeds maxNorm, scale down
    if (gradientNorm > maxNorm) {
        double scaleFactor = maxNorm / gradientNorm;
        
        // Scale down the gradient
        for (int i = 0; i < gradientTensor->numElements; i++) {
            gradientTensor->elements[i] *= scaleFactor;
        }
    }
}

Layer* initializeLayer(int numNodes, int inputDim, double (*randomFunc)(double, double), double (*activationFunction)(double), double (*activationDerivativeFunction)(double)){
  Layer* resultLayer = malloc(sizeof(Layer));
  resultLayer->numNodes = numNodes;
  resultLayer->inputDim = inputDim;

  int* shape = malloc(2 * sizeof(int));
  // CAREFUL
  shape[0] = inputDim;shape[1] = numNodes; 

  Tensor* tensor = randomTensor(shape, 2, shape[0] * shape[1], NULL);
  Tensor* gradTensor = initializeTensor(shape, 2, shape[0] * shape[1]);
  free(shape);
  resultLayer->layerTensor = tensor;
  resultLayer->gradientTensor = gradTensor;
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


Tensor* forwardMLP(MLP* mlp, Tensor* inputTensor) {
    Tensor* currentTensor = inputTensor;
    for (int i = 0; i < mlp->numLayers; i++) {
        Tensor* nextTensor = forwardLayer(mlp->layers[i], currentTensor);

        if (mlp->cacheActivations[i]) {
            freeTensor(mlp->cacheActivations[i]); // Free old cache
        }
        mlp->cacheActivations[i] = nextTensor; // Save new activation

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
  //gradientTensor = tensorTensorElementWiseMUL(gradientTensor, layer->layerTensor);

  Tensor* toFree;
  Tensor* preActivation = tensorTensorMUL(inputTensor, layer->layerTensor);
  Tensor* delta = copyTensor(outputGrad);
  Tensor* activationDerivative = copyTensor(preActivation);
  applyFuncToTensor(activationDerivative, layer->activationDerivativeFunction);

  Tensor* scaledDelta = tensorTensorElementWiseMUL(delta, activationDerivative);

  transposeTensor(inputTensor);
  Tensor* gradientTensor = tensorTensorMUL(inputTensor, scaledDelta);
  transposeTensor(inputTensor);

  toFree = layer->gradientTensor; 
  layer->gradientTensor = addTensors(layer->gradientTensor, gradientTensor);
  freeTensor(gradientTensor);
  freeTensor(toFree);

  transposeTensor(layer->layerTensor);
  Tensor* tensorInputGradient  = tensorTensorMUL(scaledDelta, layer->layerTensor);
  transposeTensor(layer->layerTensor);

  freeTensor(delta);
  freeTensor(preActivation);
  freeTensor(activationDerivative);
  freeTensor(scaledDelta);

  return tensorInputGradient;
}



void computeGradients(MLP* mlp, Tensor* inputTensor, Tensor* targetTensor){
  Tensor* output = forwardMLP(mlp, inputTensor);
  Tensor* lossGradient = computeLossGradient(output, targetTensor, mlp->costDerivativeFunction);
  Tensor* tmpGrad = lossGradient;
  Tensor* toFree;
  for(int i = mlp->numLayers - 1; i >= 0; i--){
    Tensor* prevInput = (i == 0) ? inputTensor : mlp->cacheActivations[i-1];
    toFree = tmpGrad;
    tmpGrad = backwardLayer(mlp->layers[i], prevInput, tmpGrad);
    //clipGradients(mlp->layers[i]->gradientTensor, 1.0);
    freeTensor(toFree);
  }
  freeTensor(tmpGrad);
}

Tensor* computeLossGradient(Tensor* yhat, Tensor* y, Tensor* (*costDerivativeFunction)(Tensor*, Tensor*)){
  return costDerivativeFunction(yhat, y);
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