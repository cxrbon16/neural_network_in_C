#include "nn_engine/nn_engine.h"
#include <math.h>

#define BETA_1 0.9
#define BETA_2 0.999
#define EPSILON 1e-8
#define LEARNING_RATE 0.001

Tensor* softmax(Tensor* logits) {
    // Find the maximum value in the logits
    double maxLogit = logits->elements[0];
    for (int i = 1; i < logits->numElements; i++) {
        if (logits->elements[i] > maxLogit) {
            maxLogit = logits->elements[i];
        }
    }

    // Compute the numerator (exp(logits - maxLogit)) and the denominator (sum)
    double sum = 0.0;
    double* softmaxElements = malloc(sizeof(double) * logits->numElements);
    for (int i = 0; i < logits->numElements; i++) {
        softmaxElements[i] = exp(logits->elements[i] - maxLogit); // Shift by maxLogit
        sum += softmaxElements[i];
    }

    // Normalize to get the probabilities
    for (int i = 0; i < logits->numElements; i++) {
        softmaxElements[i] /= sum;
    }

    // Create the result tensor
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
    return x > 0 ? x : 0.1 * x;
}
// Mock activation derivative
double relu_derivative(double x) {
    return x > 0 ? 1 : 0.1;
}

double calculateValidationLoss(dataPoint** testData, MLP* model, double(*lossFunction) (Tensor*, Tensor*), int testSize){
    double totalLoss = 0.0;
    for (int index = 0; index < testSize; index++){
        totalLoss += softmaxLoss(forwardMLP(model, testData[index]->X), testData[index]->Y);
    }
    return totalLoss/testSize;
}

double computeGradientsForGradientsDescent(MLP* model, dataPoint** trainData, int trainSize){
    double Loss = 0.0;
    for(int i = 0; i < trainSize; i++){
        computeGradients(model, trainData[i]->X, trainData[i]->Y);
        Loss += softmaxLoss(forwardMLP(model, trainData[i]->X), trainData[i]->Y);
    }
    return Loss;
}

double computeGradientsForSgd(MLP* model, dataPoint** trainData, int trainSize){
    double Loss = 0.0;
    int index = (int)((double)rand() / RAND_MAX * trainSize);
    // printf("%d", index);
    computeGradients(model, trainData[index]->X, trainData[index]->Y);
    return softmaxLoss(forwardMLP(model, trainData[index]->X), trainData[index]->Y);
}

Layer* initializeLayerWithAdam(int numNeurons, int inputSize, double (*activation)(double), double(*activationDerivative)(double)) {
    Layer* layer = initializeLayer(numNeurons, inputSize, NULL,  activation, activationDerivative);
    
    // Allocate memory for Adam variables
    layer->adamVars = malloc(sizeof(AdamVars));
    layer->adamVars->m = malloc(sizeof(double) * layer->layerTensor->numElements);
    layer->adamVars->v = malloc(sizeof(double) * layer->layerTensor->numElements);
    layer->adamVars->grad = malloc(sizeof(double) * layer->layerTensor->numElements);
    
    // Initialize m and v to zero
    for (int i = 0; i < layer->layerTensor->numElements; i++) {
        layer->adamVars->m[i] = 0.0;
        layer->adamVars->v[i] = 0.0;
        layer->adamVars->grad[i] = 0.0;
    }

    return layer;
}

int main(){
 srand(time(NULL));
    dataPoint** trainData;
    dataPoint** testData;
    int trainSize = 5000; 
    int testSize = 1000;
    trainData = readInput(trainSize, 785, 10, "data/train_fashion.txt");
    testData = readInput(testSize, 785, 10, "data/test_fashion.txt");


    Layer* firstHiddenLayer = initializeLayerWithAdam(64, 785, relu, relu_derivative);
    Layer* secondHiddenLayer = initializeLayerWithAdam(32, 64, relu, relu_derivative);
    Layer* thirdHiddenLayer  = initializeLayerWithAdam(10, 32, relu, relu_derivative);

    for (int i = 0; i < firstHiddenLayer->layerTensor->numElements; i++){
        printf("%.3f\n", firstHiddenLayer->layerTensor->elements[i]);;
    }
    MLP* model = malloc(sizeof(MLP));
    model->cacheActivations = malloc(sizeof(Tensor*) * 3);
    model->costFunction = softmaxLoss;
    model->costDerivativeFunction = softmaxDerivative;
    model->numLayers = 3;
    model->layers = malloc(sizeof(Layer*) * 3);
    model->layers[0] = firstHiddenLayer;
    model->layers[1] = secondHiddenLayer;
    model->layers[2] = thirdHiddenLayer;
    
    double totalLoss;
    int epoch = 20;
    double alpha = 0.05;
    Tensor* toFree;

    for (int iterateNum = 0; iterateNum < epoch * trainSize; iterateNum++) {
    totalLoss = computeGradientsForSgd(model, trainData, trainSize);
    /*
    for (int layerNo = 0; layerNo < model->numLayers; layerNo++) {
        Layer* layer = model->layers[layerNo];
        AdamVars* adamVars = layer->adamVars;

        // Update m and v for Adam
        for (int i = 0; i < layer->layerTensor->numElements; i++) {
            adamVars->grad[i] = layer->gradientTensor->elements[i];

            // Update m and v using the gradient
            adamVars->m[i] = BETA_1 * adamVars->m[i] + (1 - BETA_1) * adamVars->grad[i];
            adamVars->v[i] = BETA_2 * adamVars->v[i] + (1 - BETA_2) * adamVars->grad[i] * adamVars->grad[i];
            
            // Bias correction
            double mHat = adamVars->m[i] / (1 - pow(BETA_1, iterateNum + 1));
            double vHat = adamVars->v[i] / (1 - pow(BETA_2, iterateNum + 1));

            // Update weights with Adam
            layer->layerTensor->elements[i] -= LEARNING_RATE * mHat / (sqrt(vHat) + EPSILON);
        }
        
        zeroGradients(layer);
    }
    */
   /*
    if (iterateNum % trainSize == 0) {
        totalLoss = (double)totalLoss * (1.0);
        printf("Epoch %d - Cost: %f - Valid. Loss: %f\n", iterateNum, totalLoss, calculateValidationLoss(testData, model, softmaxLoss, testSize));
    }
    */
}
}