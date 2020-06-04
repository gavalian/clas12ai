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
    // with each read from the file we will be loadeing several
    // samples each of them having N candidates to evaluate...
    // for example if we load 2 banches, data will look like this
    // 1 0.1 .....
    // 0 0.4 .....
    // 1 0.5 .....
    // 0 0.7 .....
    // 0 0.2 .....
    // This will be 2 banches with roads = 2 in first banch and
    // roads = 3 in second banch
    int banches = read_next();
    printf("--------- event # %5d, banches = %5d\n",counter,banches);
    // itarate over each banch and find out how many candidates
    // are in the same banch
    for(int b = 0; b < banches; b++){
      // roads - is the number of candidates
      // something like:
      // 1 0.1 ......
      // 0 0.2 ......
      // 0 0.5 ......
      // 0 0.4 ......
      // in this case roads = 4
      int roads = count_roads(b);
      // create containers to hold data for 6 feature data set
      double*   roads_ptr = (double *) malloc(roads*6*sizeof(double));
      double* results_ptr = (double *) malloc(roads*sizeof(double));
      // this reads the next roads into the created array.
      read_roads(roads_ptr,roads,b);
      //----- HERE PYTHON CODE has to evaluate the
      // roads (or samples) and return an array with probabilities
      // of the labels, this loop just assigns random probabilities
      for(int r = 0; r < roads; r++) results_ptr[r] = ((double) rand())/RAND_MAX;
      // This function will be called to save probabilities to the file...
      write_roads(results_ptr,roads,b);
      free(roads_ptr); free(results_ptr);
    }
    
  }
}
