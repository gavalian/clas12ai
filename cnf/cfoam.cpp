/**
 * Event generator based on VTK file.
 * author: G.Gavalian (Dec 06 2019)
*/

#include "cfoam.h"
#include <cmath>
#include <stdlib.h>

namespace cnf {

double points::getPoint(int index, int component){
    int offset = index*ndim + component;
    return buffer[offset];
}

int points::getSize(){
  return buffer.size()/ndim;
}

void   points::getRandom(std::vector<int> &index, std::vector<double> &result){
    if(index.size()!=3){
      printf("wrong number of veticies = %lu\n",index.size());
      return;
    }
    double ax = getPoint(index[0], 0);
    double bx = getPoint(index[1], 0);
    double cx = getPoint(index[2], 0);
    double ay = getPoint(index[0], 1);
    double by = getPoint(index[1], 1);
    double cy = getPoint(index[2], 1);
    double r1 = ((double) rand())/RAND_MAX;
    double r2 = ((double) rand())/RAND_MAX;
    double sr1 = sqrt(r1);
    result.resize(2);
    result[0] = (1.0-sr1)*ax + ( sr1*(1.0-r2))*bx + (sr1*r2)*cx;
    result[1] = (1.0-sr1)*ay + ( sr1*(1.0-r2))*by + (sr1*r2)*cy;
}

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
    printf("POINTS %8d, CELLS %8lu\n",pointBuffer.getSize(),cellBuffer.size());
    for(int i = 0; i < cellIntegral.size(); i++){
      printf("%9.6f ", cellIntegral[i]);
    }
    printf("\n");
  }

  void foam::getRandom(std::vector<double> &values){
    double value = ((double) rand())/RAND_MAX;
    int bin = getBin(value);
    //printf(" value = %8.5f , bin = %5d\n",value,bin);
    pointBuffer.getRandom(cellBuffer[bin].getNodes(),values);
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
