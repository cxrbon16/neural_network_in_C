/**
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : nn_engine
 * @created     : Pazartesi Kas 25, 2024 00:01:31 +03
 */

#include "../tensor_engine/tensor_engine.h"

#ifndef NN_ENGINE_H
#define NN_ENGINE_H

typedef struct {
  Tensor* X;
  Tensor* Y;  // One-hoted labels.
} dataSet;


typedef struct {
  int numNodes;
  int inputDim;
  Tensor* layerTensor;
  Tensor* gradientTensor;
  double (*activationFunction)(double);
  double (*activationDerivativeFunction)(double);
} Layer;


typedef struct {
  int numLayers;
  Layer** layers; 
  Tensor** cacheActivations;
  // these 2 lines below can be changed. "parameters".
  double (*costFunction)(double, double);
  double (*costDerivativeFunction)(double, double);
} MLP;

Layer* initializeLayer(int numNodes, int inputDim, double (*randomFunction)(double, double), double (*activationFunction)(double), double (*activationDerivativeFunction)(double));
/*
    Bu fonksiyon, verilen parametrelerle birlikte Layer ilklendirir.

    @params:
      int numNodes: ilgil layerda kaç adet neuron/node bulunacağını belirleyen parametre.
      int inputDim: inputDim değeri bizim için her bir neuronda kaç adet değer bulunacağını belirler,
      yani bizim giriş boyutumuz aynı zamanda Sütün sayımız olacaktır.
      double(*randomFunction)(double, double): Tensördeki ağrılıkların hangi random fonksiyona göre
      dağılacağını belirleyen fonksiyondur. NULL verilirse, -1, 1 arasında uniform distr uygulanır.
      Tensor* (*activationFunction)(Tensor*): Matris çarpımı sonunda oluşacak değere uygulanacak olan
      fonksiyondur. Modelimize non-linearity katması açısından önemlidir.
    
    @outputs:
      Layer*: Yukardaki parametrelerle ilklendirilen Layer'ın referansını döner.
*/

Tensor* forwardLayer(Layer* layer, Tensor* inputTensor);
/*
    Bu fonksiyon, verilen inputTensorunu layerımızdaki weightler ile çarpıp
    sonrasında layerdaki aktivasyon fonksiyonuna sokar.

    @params:
      Layer* layer: İlgili layer.
      Tensor* inputTensor: ilgili inputTensor. 
    @outputs:
      Tensor*: Sonuç tensörü.
*/



Tensor* forwardMLP(MLP* mlp, Tensor* inputTensor);
/*
    BU fonksiyon, verilen inputTensorunu MLP'deki tüm layerlara sokar ve sonucunda bir outputTensoru döner.
    @params:
      MLP* mlp: İlgili MLP.
      Tensor* inputTensor: ilgili inputTensor. 
    @outputs:
      Tensor*: Sonuç tensörü.
*/

void computeGradients(MLP* mlp, Tensor* inputTensor, Tensor* targetTensor);
Tensor* backwardLayer(Layer* layer, Tensor* inputTensor, Tensor* outputGrad);
void backwardMLP(MLP* mlp, Tensor* input, Tensor* output);
Tensor* computeLossGradient(Tensor* yhat, Tensor* y, double (*costDerivativeFunction)(double, double));
void zeroGradients(Layer* layer);
#endif

