/**
 * Event generator based on VTK file.
 * author: G.Gavalian (Dec 06 2019)
*/

#include "cfoam.h"
#include <cmath>
#include <stdlib.h>

namespace cnf {

  void foam::init(){
     int cellBufferSize = cellBuffer.size();
     cellIntegral.resize(cellBufferSize,0.0);
     double integral = 0.0;
     for(int i = 0; i < cellBufferSize; i++){
       cellIntegral[i] = integral + cellBuffer[i].getWeight();
       integral += cellBuffer[i].getWeight();
     }
     // normalize the reimann summ.....
     double max = cellIntegral[cellBufferSize-1];
     for(int i = 0; i < cellBufferSize; i++)
        cellIntegral[i] = cellIntegral[i]/max;
  }

  void foam::show(){
    for(int i = 0; i < cellIntegral.size(); i++){
      printf("%9.6f ", cellIntegral[i]);
    }
    printf("\n");
  }

  void foam::getRandom(std::vector<double> &values){
    double value = ((double) rand())/RAND_MAX;
    int bin = getBin(value);
    printf(" value = %8.5f , bin = %5d\n",value,bin);
  }
  // This function is quick and dirty imeplementation
  // should be converted to do binary search, which
  // will be more efficient
  int  foam::getBin(double value){
    int cellIntegralSize = cellIntegral.size();
    for(int i = 0; i < cellIntegralSize; i++){
      if(cellIntegral[i]>value) return i;
    }
      return -1;
  }
}
