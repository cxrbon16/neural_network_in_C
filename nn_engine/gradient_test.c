#include "nn_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <math.h>

double compute_mse(double yHat, double y) {
    return (y - yHat) * (y- yHat);
}

double compute_mse_derivative(double yHat, double y) {
    return -2.0 * (y - yHat);
}

Tensor* softmax(Tensor* logits){
    double sum = 0.0;
    for(int i = 0; i < logits->numElements; i++)
        sum += exp(logits->elements[i]);
    double* softmaxElements = malloc(sizeof(double) * logits->numElements);
    for(int i = 0; i < logits->numElements; i++)
        softmaxElements[i] = exp(logits->elements[i]) / sum;
    int shape[] = {1, logits->numElements};
    Tensor* resultTensor = createTensor(softmaxElements, shape, 2, logits->numElements);
    return resultTensor;
}

double softmaxLoss(Tensor* logits, Tensor* targetLabels){
    double loss = 0;
    Tensor* yhat = softmax(logits);
    for(int i = 0; i < logits->numElements; i++){
        loss -= targetLabels->elements[i] * log(yhat->elements[i]);
    }
    return loss;
}
Tensor* softmaxDerivative(Tensor* logits, Tensor* y){
    Tensor* yhat = softmax(logits);
    scalarTensorMUL(y, -1.0); // this is a inplace function so we need to revert it after usage.
    Tensor* softmaxDerivative = addTensors(yhat,y);
    scalarTensorMUL(y, -1.0);
    return softmaxDerivative;

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
    /*
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

    for(int j = 0; j < mlp->numLayers; j++)
        zeroGradients(mlp->layers[j]);

    for(int i = 0; i < 1000; i++){
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
    printTensor(layerSecond->gradientTensor);
    */
}
void FULL_TEST(){
    dataPoint** trainData;
    int trainSize = 1000;
    trainData = readInput(trainSize, 785, 10, ".//data//train.txt");

    Layer* firstHiddenLayer = initializeLayer(15, 785, NULL, tanh, tanh_derivative);
    Layer* secondHiddenLayer = initializeLayer(10, 15, NULL, tanh, tanh_derivative);

    MLP* model = malloc(sizeof(MLP));
    model->cacheActivations = malloc(sizeof(Tensor*) * 2);
    model->costFunction = softmaxLoss;
    model->costDerivativeFunction = softmaxDerivative;
    model->numLayers = 2;
    model->layers = malloc(sizeof(Layer*) * 2);
    model->layers[0] = firstHiddenLayer; model->layers[1] = secondHiddenLayer;
    
    double totalLoss = 0.0;
    int epoch = 100;
    double alpha = 0.05;
    Tensor* toFree;
    for(int iterateNum = 0; iterateNum < epoch; iterateNum++){
        totalLoss = 0.0;
        for(int i = 0; i < trainSize; i++){
            computeGradients(model, trainData[i]->X, trainData[i]->Y);
            totalLoss += softmaxLoss(forwardMLP(model, trainData[i]->X), trainData[i]->Y);
        }
        for(int layerNo = 0; layerNo < model->numLayers; layerNo++){
            Layer* layer = model->layers[layerNo];
            scalarTensorMUL(layer->gradientTensor, -1.0/trainSize);
            scalarTensorMUL(layer->gradientTensor, alpha);
            toFree = layer->layerTensor;
            layer->layerTensor = addTensors(layer->layerTensor, layer->gradientTensor);
            zeroGradients(layer);
        }
        totalLoss = (double) totalLoss * (1.0/trainSize);
        printf("cost: %f\n", totalLoss);
    }
     
    // X -> reLU -> tanh -> softmax
    // (1, 785) x (785, 50) x (50, 4) -> 4 'logits' 
    // softmax(logits)
    // [a1, a2, a3, a4] that Sum(a) = 1
}

int main(){
    srand(time(NULL));
    //testBackwardLayer();
    //testMLPBackward();
    FULL_TEST();
}