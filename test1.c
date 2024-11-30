#include "nn_engine/nn_engine.h"
#include <math.h>

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

double softmaxLoss(Tensor* logits, Tensor* targetLabels) {
    const double epsilon = 1e-15;
    Tensor* yhat = softmax(logits);
    double loss = 0;
    
    for(int i = 0; i < logits->numElements; i++) {
        // Clip probability to prevent log(0)
        double clipped_prob = fmax(yhat->elements[i], epsilon);
        loss -= targetLabels->elements[i] * log(clipped_prob);
    }
    
    return loss;
}

Tensor* softmaxDerivative(Tensor* logits, Tensor* y){
    Tensor* yhat = softmax(logits);
    
    Tensor* derivative = copyTensor(yhat);
    
    for(int i = 0; i < derivative->numElements; i++) {
        derivative->elements[i] -= y->elements[i];
    }
    
    return derivative;
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

double calculateValidationLoss(dataPoint** testData, MLP* model, double(*lossFunction) (Tensor*, Tensor*), int testSize){
    double totalLoss = 0.0;
    for (int index = 0; index < testSize; index++){
        totalLoss += softmaxLoss(forwardMLP(model, testData[index]->X), testData[index]->Y);
    }
    return totalLoss/testSize;
}

MLP* initializeMLP(int numLayers, int* layerSizes, double (**activations)(double), double (**activationDerivatives)(double)) {
    MLP* model = malloc(sizeof(MLP));
    model->numLayers = numLayers;
    model->layers = malloc(numLayers * sizeof(Layer*));
    model->cacheActivations = malloc(numLayers * sizeof(Tensor*));

    for (int i = 0; i < numLayers; i++) {
        model->layers[i] = initializeLayer(layerSizes[i+1], layerSizes[i], NULL, activations[i], activationDerivatives[i]);
    }

    model->costFunction = softmaxLoss;
    model->costDerivativeFunction = softmaxDerivative;

    return model;
}

void gradientDescent(MLP* model, double alpha, int trainSize) {
    for (int layerNo = 0; layerNo < model->numLayers; layerNo++) {
        Layer* layer = model->layers[layerNo];

        // Scale gradients by the learning rate and negative step size
        scalarTensorMUL(layer->gradientTensor, -1.0 / trainSize);
        scalarTensorMUL(layer->gradientTensor, alpha);

        // Update weights
        Tensor* updatedWeights = addTensors(layer->layerTensor, layer->gradientTensor);

        // Free the old tensor and replace it with updated weights
        freeTensor(layer->layerTensor);
        layer->layerTensor = updatedWeights;

        // Reset gradients for the next iteration
        zeroGradients(layer);
    }
}

int main() {
    srand(time(NULL));

    dataPoint** trainData;
    dataPoint** testData;
    int trainSize = 5000; 
    int testSize = 1000;
    trainData = readInput(trainSize, 785, 10, "data/train_fashion.txt");
    testData = readInput(testSize, 785, 10, "data/test_fashion.txt");

    int numLayers = 2;
    int layerSizes[] = {785, 128, 10}; // Input layer, hidden layer, output layer
    double (*activations[])(double) = {relu, tanh};
    double (*activationDerivatives[])(double) = {relu_derivative, tanh_derivative};

    MLP* model = initializeMLP(numLayers, layerSizes, activations, activationDerivatives);

    int epoch = 1000;
    double alpha = 0.005;

    for (int iterateNum = 0; iterateNum < epoch; iterateNum++) {
        double totalLoss = 0.0;
        for (int i = 0; i < trainSize; i++) {
            computeGradients(model, trainData[i]->X, trainData[i]->Y);
            totalLoss += softmaxLoss(forwardMLP(model, trainData[i]->X), trainData[i]->Y);
        }

        gradientDescent(model, alpha, trainSize);

        if (iterateNum % 10 == 0) {
            double validationLoss = calculateValidationLoss(testData, model, softmaxLoss, testSize);
            printf("Epoch %d - Cost: %f - Valid. Loss: %f\n", iterateNum, totalLoss / trainSize, validationLoss);
        }
    }
}
