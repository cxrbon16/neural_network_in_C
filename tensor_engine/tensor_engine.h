/**
 * @author      : ayganyavuz (ayganyavuz@ayganyavuzEXCALIBURG770)
 * @file        : tensor_engine
 * @created     : Cumartesi Kas 23, 2024 18:41:52 +03
 */

#ifndef TENSOR_ENGINE_H
#define TENSOR_ENGINE_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  double* elements;
  int* shape;
  int numShape;
  int numElements;
} Tensor;

Tensor* initializeTensor(int* shape, int numShape, int numElements);
/*
    Bu fonksiyon, ilklendirilmek istenen Tensor'un ilgili parametrelerini 
    kullanarak  ilklendirilmesini sağlar.

    @params: 
      int* shape: ilgili Tensorun boyutlarının büyüklüğü ifade eder.
      int numShape: ilgili Tensorun kaç boyutlu olduğunu ifade eder.
      int numElements: ilgili Tensorun kaç elemandan oluştuğunu gösterir.

    @output:
      Tensor*: ilklendirilen Tensorun pointerını döner. bknz: "tensor_engine.h/Tensor"
*/


Tensor* randomTensor(int* shape, int numShape, int numElements, double (*randomFunc)(double, double));
/*
    Bu fonksiyon, ilklendirilmek istenen Tensor'un ilgili parametrelerini
    ve elemanlarının hangi random fonksiyona göre belirleneceğini ifade eden
    bir fonksiyon pointer'ını kullanarak ilgili tensorun random fonksiyona göre
    ilklendirilmesini sağlar.

    @params: 
      int* shape: ilgili Tensorun boyutlarının büyüklüğü ifade eder.
      int numShape: ilgili Tensorun kaç boyutlu olduğunu ifade eder.
      int numElements: ilgili Tensorun kaç elemandan oluştuğunu gösterir.
      double (*randomFunc)(double, double): Tensorun elemanlarının hangi fonksiyona göre ilklendirileceğini gösterir, NULL olarak verilirse, 
      default bir random fonksiyon kullanır. bknz: "defaultRandomElementWise"
      ilgili randomFunc lowerBound ve upperBound olmak üzere iki double değer alır.

    @output:
      Tensor*: ilklendirilen Tensorun pointerını döner. bknz: "tensor_engine.h/Tensor"
*/

Tensor* tensorTensorMUL(Tensor* tensor, Tensor* tensor2);
/*
    Bu fonksiyon verilen iki tensoru "matris çarpımı" ile çarpar. !"Verilen tensorlerin
    yalnızca 2 adet boyutu olmalıdır ve boyutların birbirleriyle matris çarpımı
    yapılacak şekilde olması gerekir." Tensorlerin boyutlarını değiştirmek için
    "tensorChangeShape" fonksiyonu kullanılmalıdır. 

    @params: 
      Tensor* tensor: Çarpımın solunda olacak olan tensor'dur. Matris çarpımında işlemin 
      yer değiştirme/commutative özelliği olmadığı unutlmamaıldırr.
      Yani AXB eşit değildir BXA. Ayrıca tensor1.shape[1] = tensor2.shape[0] sağlanmalıdır.
      Tensor* tensor2: Çarpımın sağında olacak olan tensor'dur.

    @output:
      Tensor*: sonuç Tensorunun pointerını döner. bknz: "tensor_engine.h/Tensor"
*/

double getElementFromTensor(Tensor* tensor, int* dim, int dimSize);
/*
    Bu fonskiyon ilgili tensorun ilgili konumundaki elemanını döner. 

    @params: 
      Tensor* tensor: İçindeki elemanın konumunu merak ettiğimiz tensor. 
      int* dim: Tensorun içindeki ilgilendiğimiz eleman.

    @output:
      double: ilgili elemanın değerini döner.
*/

void tensorChangeShape(Tensor* tensor, int* newShape, int newNumShape);
/*
    Bu fonksiyon, ilgili tensorun shape'ini değiştirmeye yarar. 
    İşlemin gerçekleşmesi için eleman sayılarının tutması gerekir.

    @params: 
      Tensor* tensor: Shape'ini değiştirmek istediğimiz tensor.
      int* newShape: İstenen shape.
      int newNumShape: İstenen shape'in kaç boyutlu olduğu.

    @output:
      --
      'Bu fonksiyon in-place çalışır, herhangi bir return değeri yoktur.
*/
void transposeTensor(Tensor* tensor);
/*
    Bu fonksiyon, ilgili tensorun, transpose'unu almayı sağlar. Ancak yalnızca 2 boyuta sahip 
    tensorlar, matrisler, ile çalışır. 2'den fazla boyuta sahip olan bir tensorun transposeunu almak için
    önce "tensorChangeShape" fonksiyonu çağrılmalıdır. 

    @params: 
      Tensor* tensor: Transpose'unu almak istediğimiz tensor.

    @output:
      --
      'Bu fonksiyon in-place çalışır, herhangi bir return değeri yoktur.
*/
void scalarTensorMUL(Tensor* tensor, double scalar);
/*
    Bu fonksiyon, tensor'deki tüm elemanları bir scalar ile çarpar.
    @params: 
      Tensor* tensor: Elemenlarını çarpmak istediğimiz tensor.
      double scalar: çarpma katsayısı.

    @output:
      --
      'Bu fonksiyon in-place çalışır, herhangi bir return değeri yoktur.
*/
void applyFuncToTensor(Tensor* tensor, double (*func)(double));
/*
    Bu fonksiyon, tensor'deki tüm elemanlara func fonksiyonunu uygular.
    @params: 
      Tensor* tensor: Elemenlarına fonksiyonu uygulamak istediğimiz tensor.
      double (*func)(double): Uygulayacağımız fonksiyon, ilgili fonksiyon double bir değer alıp
      double bir değer dönmelidir.

    @output:
      --
      'Bu fonksiyon in-place çalışır, herhangi bir return değeri yoktur.
*/

Tensor* addTensors(Tensor* tensor, Tensor* otherTensor);
/*
    Bu fonksiyon, parametre olarak verilen iki tensörü toplayıp sonucu yeni bir tensör olarak döner.
    @params: 
      Tensor* tensor: Tensör toplamının birinci operand'ı. 
      Tensor* otherTensor: Tensör toplamının ikinci operand'ı. 

    @output:
      Tensor*: Sonuç olarak çıkan tensör dönülür.
*/

void writeTensorToFile(Tensor* tensor, FILE* filePath);
void freeTensor(Tensor* tensor);
/*
    Bu fonksiyon, parametre olarak verilen tensör referansındaki struct'ı ve içindeki diğer malloc edilen referansları free'ler. 
    @params: 
      Tensor* tensor: Free edilecek tensör referansı. 

    @output:
      --
      Bu fonksiyon, herhangi bir değer dönmez.
*/

Tensor* copyTensor(Tensor* tensor);
void printTensor(Tensor* tensor);
Tensor* tensorTensorElementWiseMUL(Tensor* tensor, Tensor* otherTensor);
Tensor* createTensor(double* elements, int* shape, int numShape, int numElements);
#endif

