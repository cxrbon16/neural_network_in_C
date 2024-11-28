#include "nn_engine/nn_engine.h"
#include <math.h>
#define TRAIN_SIZE 50

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

double randomFunc(double x, double y){
  double random_value = (rand() / (double)RAND_MAX)  * sqrt(2.0 / TRAIN_SIZE);
}

int main(){
    srand(time(NULL));
    dataPoint** trainData;
    int trainSize = 500;
    trainData = readInput(trainSize, 785, 10, "data/train_fashion.txt");

    Layer* firstHiddenLayer = initializeLayer(1200, 785, NULL, relu, relu_derivative);
    Layer* secondHiddenLayer = initializeLayer(10, 1200, NULL, relu, relu_derivative);

    MLP* model = malloc(sizeof(MLP));
    model->cacheActivations = malloc(sizeof(Tensor*) * 2);
    model->costFunction = softmaxLoss;
    model->costDerivativeFunction = softmaxDerivative;
    model->numLayers = 2;
    model->layers = malloc(sizeof(Layer*) * 2);
    model->layers[0] = firstHiddenLayer; model->layers[1] = secondHiddenLayer;
    
    double totalLoss = 0.0;
    int epoch = 100;
    double alpha = 0.005;
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
            freeTensor(toFree);
            zeroGradients(layer);
        }
        totalLoss = (double) totalLoss * (1.0/trainSize);
        printf("cost: %f\n", totalLoss);
    }
    for(int i = 0; i < 10; i++){
        printf("predict: \n");
        printTensor(softmax(forwardMLP(model, trainData[i]->X)));
        printf("target: \n");
        printTensor(trainData[i]->Y);
    }
}