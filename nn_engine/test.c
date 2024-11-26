/**
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : test
 * @created     : Pazartesi Kas 25, 2024 00:03:35 +03
 */
#include "nn_engine.h"
#include <stdio.h>
#include <math.h>

double tanh_x(double x){
    return tanh(x);
}

int main(){
    Layer* layer1 = initializeLayer(6, 100, NULL, tanh_x);
    Layer* layer2 = initializeLayer(2, 6, NULL, tanh_x);
    int shapeInput[] = {5, 100};
    Tensor* inputTensor = randomTensor(shapeInput, 2, 500, NULL);

    // TODO: random tensor'un parametreleri daha sade hale getirilmeli.
    

    // Pay attention to memory leak scenarios for 4 consecutive lines of code.!
    Tensor* resultTensor = forwardLayer(layer1, inputTensor);
    for (int i = 0; i < resultTensor->numShape; i++){
        printf("%d \n", resultTensor->shape[i]);
    }
    Tensor* tempTensor = forwardLayer(layer2, resultTensor);
    resultTensor = tempTensor;

    for (int i = 0; i < resultTensor->numShape; i++){
        printf("%d \n", resultTensor->shape[i]);
    }
    for(int i = 0; i < resultTensor->numElements; i++){
        printf("%f \n", resultTensor->elements[i]);
    }
    /*
    freeTensor(resultTensor);
    freeTensor(inputTensor);
    freeTensor(layer1->layerTensor);
    freeTensor(layer2->layerTensor);

    free(layer1);
    free(layer2);
    */
}
