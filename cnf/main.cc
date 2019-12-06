#include "cfoam.h"


void foam_debug();

int main(){
  printf("welcome to foam simulator.....\n");

  foam_debug();
  return 1;
}


void foam_debug(){
  cnf::foam foam;

  for(int i = 0; i < 10; i++){
    cnf::cell c;
    c.setWeight(0.05+0.1*i);
    foam.addCell(c);
  }
  foam.init();
  foam.show();
  std::vector<double> values;
  for(int i = 0; i < 20; i++){
    foam.getRandom(values);
  }
}
