#include <stdlib.h>
#include <stdio.h>

extern "C" {
  void  open_file(const char *filename);
  int   read_next();
  int   count_roads( int banch);
  void  read_roads ( double *roads, int nroads, int banch);
  void  write_roads( double *roads_results, int nroads, int banch);
}

int main(){

  char dataFile[128] = "/somepath/somefile.txt";

  open_file(dataFile);

  int counter = 0;

  while(counter<10){
    counter++;
    int banches = read_next();
    printf("--------- event # %5d, banches = %5d\n",counter,banches);
    for(int b = 0; b < banches; b++){

      int roads = count_roads(b);
      
      double*   roads_ptr = (double *) malloc(roads*6*sizeof(double));
      double* results_ptr = (double *) malloc(roads*sizeof(double));
      read_roads(roads_ptr,roads,b);

      for(int r = 0; r < roads; r++) results_ptr[r] = rand()/RAND_MAX;
      write_roads(results_ptr,roads,b);
      free(roads_ptr); free(results_ptr);
    }
    
  }
}
