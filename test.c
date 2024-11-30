#include "nn_engine/nn_engine.h"
#include <math.h>

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

double computeAccuracy(MLP* model, dataPoint** testData, int testSize){
    Tensor* logits;
    double max;
    int maxIndex = 0;
    int correct = 0;
    int incorrect = 0;
    for(int i = 0; i < testSize; i++){
        logits = forwardMLP(model, testData[i]->X);
        max = logits->elements[0];
        maxIndex = 0;
        for(int Index = 0; Index < logits->numElements; Index++){
            if (logits->elements[Index] > max){
                max = logits->elements[Index];
                maxIndex = Index;
            }
        } 
        for(int j = 0; j < testData[i]->Y->numElements; j++){
            if(testData[i]->Y->elements[j] == 1.0 && maxIndex == j){
               correct += 1; 
               continue;
            }
        }
        incorrect += 1;
    }
    return (double) correct * 100 / (double) (testSize);
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


int main(){
    srand(time(NULL));
    dataPoint** trainData;
    dataPoint** testData;
    int trainSize = 60000; 
    int testSize = 10000;
    trainData = readInput(trainSize, 785, 10, "data/train.txt");
    testData = readInput(testSize, 785, 10, "data/test.txt");


    Layer* firstHiddenLayer = initializeLayer(64, 785, NULL, tanh, tanh_derivative);
    Layer* secondHiddenLayer = initializeLayer(10, 64, NULL, tanh, tanh_derivative);

    for ( int i = 0; i < firstHiddenLayer->layerTensor->numElements; i++){
        printf("%.3f\n", firstHiddenLayer->layerTensor->elements[i]);;
    }
    MLP* model = malloc(sizeof(MLP));
    model->cacheActivations = malloc(sizeof(Tensor*) * 2);
    model->costFunction = softmaxLoss;
    model->costDerivativeFunction = softmaxDerivative;
    model->numLayers = 2;
    model->layers = malloc(sizeof(Layer*) * 2);
    model->layers[0] = firstHiddenLayer;
    model->layers[1] = secondHiddenLayer;
    
    double totalLoss;
    int epoch = 4;
    double alpha = 0.005;
    Tensor* toFree;

    Tensor* output = forwardMLP(model, trainData[0]->X);
    printTensor(output);
    Tensor* lossGradient = computeLossGradient(output, trainData[0]->Y, model->costDerivativeFunction);
    printf("\n Loss gradients: \n");
    printTensor(lossGradient);
   // printf("\n preactivation values: \n");
   // printTensor(forwardLayer(firstHiddenLayer, trainData[0]->X));
   // printf("\n Hidden gradients: \n");
   // computeGradients(model, trainData[0]->X, trainData[0]->Y);
   // printTensor(firstHiddenLayer->gradientTensor);
    for(int iterateNum = 0; iterateNum < epoch * trainSize + 1; iterateNum++){
        totalLoss = computeGradientsForSgd(model, trainData, trainSize);

        for(int layerNo = 0; layerNo < model->numLayers; layerNo++){
            Layer* layer = model->layers[layerNo];
            // scalarTensorMUL(layer->gradientTensor, -1.0/trainSize); // GD
            scalarTensorMUL(layer->gradientTensor, -1.0); // SGD
            //printf("Layer%d \n \n: ", layerNo+1);
            //printTensor(layer->gradientTensor);
            scalarTensorMUL(layer->gradientTensor, alpha);
            
            Tensor* toFree = layer->layerTensor;
            layer->layerTensor = addTensors(layer->layerTensor, layer->gradientTensor);
            freeTensor(toFree);
            zeroGradients(layer);
        }

        if(iterateNum % 10000  == 0){
            totalLoss = (double) totalLoss * (1.0);
            printf("Epoch %d - Cost: %f - Valid. Loss: %f\n Accuracy: %f\n",
             iterateNum / trainSize, totalLoss, calculateValidationLoss(testData, model, softmaxLoss, testSize), computeAccuracy(model, testData, testSize));
            /*
            Tensor* output = forwardMLP(model, trainData[iterateNum/trainSize]->X);
            output = softmax(output);
            printTensor(output);
            */
        }
    }
}