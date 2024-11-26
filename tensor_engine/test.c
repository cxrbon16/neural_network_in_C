/**
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : test
 * @created     : Cumartesi Kas 23, 2024 19:45:57 +03
 */

#include "tensor_engine.h"

int main(){

  srand(time(0));
  int* dim = malloc(2 * sizeof(int));
  dim[0] = 2; dim[1] = 2;
  Tensor* newTensor = randomTensor(dim, 2, 4, NULL);
  Tensor* newTensor2 = randomTensor(dim, 2, 4, NULL);
  
  Tensor* xTensor = tensorTensorMUL(newTensor, newTensor2);

  for(int i = 0; i < newTensor->numElements; i++){
    printf("%f\n", newTensor->elements[i]);
  }
  for(int i = 0; i < newTensor2->numElements; i++){
    printf("%f\n", newTensor2->elements[i]);
  }

  for(int i = 0; i < xTensor->numElements; i++){
    printf("%f\n", xTensor->elements[i]);
  }
  int dimX[] = {0, 1};
  printf("xTensor[0, 1] = %f\n", getElementFromTensor(xTensor, dimX, 2));
  int* newDim = malloc(sizeof(int));
  //newDim[0] = 4;
  //tensorChangeShape(xTensor, newDim, 1);
  printf("Before transpose\n");
  for(int i = 0; i < xTensor->numElements; i++){
    printf("%f\n", xTensor->elements[i]);
  }
  transposeTensor(xTensor);
  printf("After transpose\n");
  for(int i = 0; i < xTensor->numElements; i++){
    printf("%f\n", xTensor->elements[i]);
  }
  return 0;
}
