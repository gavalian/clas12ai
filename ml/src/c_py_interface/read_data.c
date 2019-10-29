//#####################################################
//# example c-code for reading samples and passing them
//# to python script to be evaluated and returning
//# results to the c program for further processing.
//# author: gavalian (2019)
//#####################################################

#include <stdlib.h>
#include <stdio.h>
#define ARRAY_SIZE 100*4032



extern "C" {
  void  open_file(const char *filename);
  int   read_next();
  int   count_roads( int banch);
  void  read_roads ( double *roads, int nroads, int banch);
  void  write_roads( double *roads_results, int nroads, int banch);
}

//====================================================
//= This routine will open given file to provide
//= samples of track candidates to the routines.
//====================================================
void open_file(const char *filename){
  printf("---> some file was opened with name : %s\n",filename);
}
//========================================================
//= Read next bunch of track candidates and return
//= the number of candidates that were read.
//= if end of file war reached, return value will be -1
//= otherwise number of bunches is returned (0 is also
//= valid return value).
//========================================================
int read_next(){
  return rand()%6+3;
}

int count_roads(int banch){
  return rand()%3+4;
}

void  read_roads ( double *roads, int nroads, int banch){
  int nfeatures = 6;
  for(int i = 0; i < nroads*nfeatures; i++){
    roads[i] = ((double)rand())/RAND_MAX;
  }
}

void  write_roads( double *roads_results, int nroads, int banch){
  printf("banch %5d : results = ",banch);
  for(int i = 0; i < nroads; i++){
    printf(" %6.3f ",roads_results[i]);
  }
  printf("\n");
}
