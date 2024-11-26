#include "nn_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Define Tensor structure and auxiliary functions here (e.g., initialization, free, etc.)
#include <math.h>

double compute_mse(double yHat, double y) {
    return (y - yHat) * (y- yHat);
}

double compute_mse_derivative(double yHat, double y) {
    return -2.0 * (y - yHat);
}

double tanh_derivative(double x){
    return 1 - tanh(x) * tanh(x);
}
// Mock activation function (ReLU)
double relu(double x) {
    return x > 0 ? x : 0;
}

// Mock activation derivative
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Helper to initialize a tensor
Tensor* createTensor(double* elements, int* shape, int numShape, int numElements) {
    Tensor* tensor = malloc(sizeof(Tensor));
    tensor->elements = malloc(sizeof(double) * numElements);
    memcpy(tensor->elements, elements, sizeof(double) * numElements);

    tensor->shape = malloc(sizeof(int) * numShape);
    memcpy(tensor->shape, shape, sizeof(int) * numShape);

    tensor->numShape = numShape;
    tensor->numElements = numElements;
    return tensor;
}


// Test Function
void testBackwardLayer() {
    // Define layer structure
    Layer layer;
    layer.numNodes = 2;

    // Initialize inputTensor
    double inputElements[] = {1.5, 2.0};
    int inputShape[] = {1, 2};
    Tensor* inputTensor = createTensor(inputElements, inputShape, 2, 2);

    // Initialize layer weights (layerTensor)
    double weightElements[] = {0.5, 0.2, 0.1, 0.8};
    int weightShape[] = {2, 2};
    layer.layerTensor = createTensor(weightElements, weightShape, 2, 4);

    // Initialize gradient storage
    layer.gradientTensor = NULL;

    // Set activation function and derivative
    layer.activationFunction = relu;
    layer.activationDerivativeFunction = relu_derivative;

    // Initialize outputGrad (DLoss/Doutput)
    double gradElements[] = {0.5, -1.2};
    int gradShape[] = {1, 2};

    Tensor* outputGrad = createTensor(gradElements, gradShape, 2, 2);

    // Call backwardLayer
    Tensor* tensorInputGradient = backwardLayer(&layer, inputTensor, outputGrad);
    
    // Print results
    printf("Updated layer gradient tensor:\n");
    printTensor(layer.gradientTensor); // Implement a function to print tensors.

    printf("Input gradient tensor:\n");
    printTensor(tensorInputGradient); // Implement a function to print tensors.
/*
    // Clean up
    freeTensor(inputTensor);
    freeTensor(outputGrad);
    freeTensor(tensorInputGradient);
    freeTensor(layer.layerTensor);
    freeTensor(layer.gradientTensor);
*/
}

void testMLPBackward(){
    int inputShape[] = {1, 20};
    int outputShape[] = {1, 1};
    Tensor* inputTensor = randomTensor(inputShape, 2, inputShape[0] * inputShape[1], NULL);
    Tensor* outputTensor = randomTensor(outputShape, 2, 1, NULL);

    Layer* layer = initializeLayer(30, 20, NULL, tanh, tanh_derivative);
    Layer* layerSecond = initializeLayer(1, 30, NULL, tanh, tanh_derivative);

    MLP* mlp = malloc(sizeof(MLP));
    mlp->numLayers = 2;
    mlp->layers = malloc(sizeof(Layer*) * mlp->numLayers);
    mlp->layers[0] = layer; mlp->layers[1] = layerSecond;
    mlp->cacheActivations = malloc(sizeof(Tensor*) * 2);

    mlp->costFunction = compute_mse;
    mlp->costDerivativeFunction = compute_mse_derivative;

    computeGradients(mlp, inputTensor, outputTensor);
    printTensor(layerSecond->gradientTensor);

    for(int i = 0; i < 100; i++){
        computeGradients(mlp, inputTensor, outputTensor);
        for(int j = 0; j < mlp->numLayers; j++){
            Layer* currLayer = mlp->layers[j];
            scalarTensorMUL(currLayer->gradientTensor, -0.05);
            currLayer->layerTensor = addTensors(currLayer->layerTensor, currLayer->gradientTensor);
            zeroGradients(currLayer);
        }
    }

    computeGradients(mlp, inputTensor, outputTensor);
    printf("\n after gd");
    printTensor(layer->gradientTensor);
}

int main(){
    srand(time(NULL));
    //testBackwardLayer();
    testMLPBackward();
    //int shape[] = {10, 10};
    //Tensor* newTensor = randomTensor(shape, 2, 100, NULL);
    //printTensor(newTensor);
}