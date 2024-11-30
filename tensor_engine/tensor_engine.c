/*
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : tensor_engine
 * @created     : Cumartesi Kas 23, 2024 18:51:28 +03
 */

#include "tensor_engine.h"

double defaultRandomElementWise(double x, double y){ // XAVIER INITIALIZATION
  double bound = sqrt(6.0f / (x + y));
  double tmp = (float)rand() / RAND_MAX * 2 * bound - bound;
  return tmp * 3;
}


Tensor* initializeTensor(int* shape, int numShape, int numElements){
  Tensor* resultTensor = malloc(sizeof(Tensor));
  double* elements = malloc(sizeof(double) * numElements);

  for(int place = 0; place < numElements; place++){
    elements[place] = 0.0; 
  }

  int* copyShape = malloc(numShape * sizeof(int));
  for(int i = 0; i < numShape; i++){
    copyShape[i] = shape[i];
  }
  resultTensor->numShape = numShape;
  resultTensor->numElements = numElements;
  resultTensor->elements = elements;
  resultTensor->shape = copyShape;
  
  return resultTensor;

}
Tensor* randomTensor(int* shape, int numShape, int numElements, double (*randomFunc)(double, double)){
  Tensor* resultTensor = malloc(sizeof(Tensor));
  double* elements = malloc(sizeof(double) * numElements);

  if(randomFunc == NULL){ 
    randomFunc = defaultRandomElementWise;
  }

  for(int place = 0; place < numElements; place++){
    elements[place] = randomFunc(shape[0], shape[1]);
  }

  int* copyShape = malloc(numShape * sizeof(int));
  for(int i = 0; i < numShape; i++){
    copyShape[i] = shape[i];
  }
  resultTensor->numShape = numShape;
  resultTensor->numElements = numElements;
  resultTensor->elements = elements;
  resultTensor->shape = copyShape;
  
  return resultTensor;
}

Tensor* tensorTensorMUL(Tensor* tensor, Tensor* tensor2){
  if(tensor->numShape != 2 || tensor2->numShape != 2){
    printf("\nTo multiplicate a tensor, you should transform it to a Matrix first.!!!!!!");
    printf("%d, %d", tensor->shape[0], tensor->shape[1]);
    printf("%d, %d", tensor2->shape[0], tensor2->shape[1]);
    return NULL;
  }
  else if(tensor->shape[1] != tensor2->shape[0]){
    printf("\n Shapes of tensors are inappropriate for a multiplication.!!!!!");
    printf("%d, %d", tensor->shape[0], tensor->shape[1]);
    printf("%d, %d", tensor2->shape[0], tensor2->shape[1]);
    return NULL;
  }

  Tensor* resultTensor = malloc(sizeof(Tensor));
  int numElements = tensor->shape[0] * tensor2->shape[1];
  double* resultElements = malloc(sizeof(double) * numElements);
  double intermediateSum = 0;
  for(int i = 0; i < tensor->shape[0]; i++){
    for (int k = 0; k < tensor2->shape[1]; k++){
      intermediateSum = 0; 
      for(int j = 0; j < tensor->shape[1]; j++){
        intermediateSum += 
        tensor->elements[i*tensor->shape[1] + j] 
        *
        tensor2->elements[j*tensor2->shape[1] + k];
      }
      resultElements[i*tensor2->shape[1] + k] = intermediateSum;
    }
  }

  int* newDim = malloc(2 * sizeof(int));
  newDim[0] = tensor->shape[0]; newDim[1] = tensor2->shape[1];
  resultTensor->elements = resultElements;
  resultTensor->numElements = numElements;
  resultTensor->shape = newDim;
  resultTensor->numShape = 2;

  return resultTensor;
}

double getElementFromTensor(Tensor* tensor, int* dim, int dimSize){
  int numShape = tensor->numShape;
  if(numShape != dimSize){
    printf("\nInput number of dim is not equal to number of dimensions of tensor.");
    return 0.0;
  }

  int* carpanlar = malloc(sizeof(int) * numShape);
  int currCarpan = 1;
  carpanlar[numShape-1] = 1;
  for(int i = numShape - 2; i >= 0; i--){
    currCarpan *= tensor->shape[i];
    carpanlar[i] = currCarpan;
  }
  int index = 0;
  for(int i = 0; i < numShape; i++){
    index += dim[i] * carpanlar[i];
  }
  free(carpanlar);
  return tensor->elements[index];
}

void tensorChangeShape(Tensor* tensor, int* newShape, int newNumShape){
  int inputElementNumber = 1;
  for(int i = 0; i < newNumShape; i++){
    inputElementNumber *= newShape[i];
  }
  if(inputElementNumber != tensor->numElements){
    printf("\nShape is unappropiate");
    return;
  }

  int* newShapeCopy = malloc(newNumShape * sizeof(int));
  for(int i = 0; i < newNumShape; i++){
    newShapeCopy[i] = newShape[i];
  }
  free(tensor->shape);
  tensor->shape = newShapeCopy;
  tensor->numShape = newNumShape;
}

void transposeTensor(Tensor* tensor){
  if(tensor->numShape != 2){
    printf("\nTranspose can only be applied to tensors that have 2 dimensions, a Matrix.");
    return;
  }
  int* newShape = malloc(2 * sizeof(int));
  newShape[0] = tensor->shape[1]; 
  newShape[1] = tensor->shape[0]; 

  double* newElements = malloc(tensor->numElements * sizeof(double));
  for(int i = 0; i < tensor->numElements; i++){
    int colIndex = i % tensor->shape[1];
    int rowIndex = i / tensor->shape[1];
    newElements[colIndex * newShape[1] + rowIndex] = tensor->elements[i];
  }
  free(tensor->elements);
  tensor->elements = newElements;
  tensor->shape = newShape;
}

void scalarTensorMUL(Tensor* tensor, double scalar){
  for(int i = 0; i < tensor->numElements; i++)
    tensor->elements[i] *= scalar;
}

void applyFuncToTensor(Tensor* tensor, double (*func)(double)){
  for(int i = 0; i < tensor->numElements; i++){
    tensor->elements[i] = func(tensor->elements[i]);
  }
}


Tensor* addTensors(Tensor* tensor, Tensor* otherTensor){
  if(tensor->numShape != otherTensor->numShape){
    printf("/n Tensors numShape is different!");
    return NULL;
  }
  for(int i = 0; i < tensor->numShape; i++){
    if(tensor->shape[i] != otherTensor->shape[i]){
      printf("%d", i);
      printf("%d", tensor->shape[i]);
      printf("%d", otherTensor->shape[i]);
      printf("Tensor1 shape: [%d, %d]: Tensor2 shape: [%d, %d]\n", tensor->shape[0], tensor->shape[1], otherTensor->shape[0], otherTensor->shape[1]);
      printTensor(tensor);
      printTensor(otherTensor);
      printf("\n Tensors shape is different!");
      return NULL;
    }
  }
  int* copyShape = malloc(tensor->numShape * sizeof(int));
  for(int i = 0; i < tensor->numShape; i++)
    copyShape[i] = tensor->shape[i];


  double* elements = malloc(tensor->numElements * sizeof(double));
  Tensor* resultTensor = malloc(sizeof(Tensor));

  for(int i = 0; i < tensor->numElements; i++){
    elements[i] = tensor->elements[i] + otherTensor->elements[i];
  }
  

  resultTensor->elements = elements;
  resultTensor->shape = copyShape;
  resultTensor->numElements = tensor->numElements;
  resultTensor->numShape = tensor->numShape;

  return resultTensor;

}

Tensor* copyTensor(Tensor* tensor){
  Tensor* resultTensor = malloc(sizeof(tensor));
  int* copyShape = malloc(sizeof(int) * 2);
  for(int i = 0; i < tensor->numShape; i++){
    copyShape[i] = tensor->shape[i];
  }
  double* elements = malloc(sizeof(double) * tensor->numElements);
  for(int i = 0; i < tensor->numElements; i++){
    elements[i] = tensor->elements[i];
  }

  resultTensor->shape = copyShape;
  resultTensor->elements = elements;
  resultTensor->numShape = tensor->numShape;
  resultTensor->numElements = tensor->numElements;

  return resultTensor;
}

void freeTensor(Tensor* tensor){
  free(tensor->elements);
  free(tensor->shape);
  free(tensor);
}

// Helper to calculate the offset for multidimensional tensors
int calculateOffset(int* shape, int numShape, int* indices) {
    int offset = 0;
    int multiplier = 1;

    for (int i = numShape - 1; i >= 0; i--) {
        offset += indices[i] * multiplier;
        multiplier *= shape[i];
    }

    return offset;
}

// Recursive function to print tensors in a structured format
void recursivePrint(double* elements, int* shape, int numShape, int dimension, int* indices) {
    if (dimension == numShape - 1) {
        // Print the innermost dimension
        printf("[");
        for (int i = 0; i < shape[dimension]; i++) {
            indices[dimension] = i;
            int offset = calculateOffset(shape, numShape, indices);
            printf("%.4f", elements[offset]);
            if (i < shape[dimension] - 1) printf(", ");
        }
        printf("]");
        return;
    }

    printf("[");
    for (int i = 0; i < shape[dimension]; i++) {
        indices[dimension] = i;
        recursivePrint(elements, shape, numShape, dimension + 1, indices);
        if (i < shape[dimension] - 1) printf(", ");
    }
    printf("]");
}

// Main function to print the tensor
void printTensor(Tensor* tensor) {
    if (!tensor || !tensor->elements || !tensor->shape) {
        printf("Invalid tensor.\n");
        return;
    }

    int* indices = calloc(tensor->numShape, sizeof(int));
    recursivePrint(tensor->elements, tensor->shape, tensor->numShape, 0, indices);
    free(indices);
    printf("\n");
}

Tensor* tensorTensorElementWiseMUL(Tensor* tensor, Tensor* otherTensor){
  if(tensor->numElements != otherTensor->numElements){
    printf("noooo.noo.no");
    return NULL;
  }
  Tensor* resultTensor = copyTensor(tensor);

  for (int i = 0; i < tensor->numElements; i++){
    resultTensor->elements[i] = tensor->elements[i] * otherTensor->elements[i];
  }

  return resultTensor;
}

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