//#####################################################
//# example c-code for reading samples and passing them
//# to python script to be evaluated and returning
//# results to the c program for further processing.
//# author: gavalian (2019)
//#####################################################

#include <stdlib.h>
#include <stdio.h>
#include "reader.h"
#include "writer.h"
#include "dcana.h"

#define ARRAY_SIZE 100*4032

hipo::reader       reader;
hipo::writer       writer;
hipo::event        event;
hipo::dictionary   factory;
hipo::bank        *hitsBank;
hipo::bank        *trackBank;
hipo::bank        *nnBank;
hipo::bank        *dc_tdc;
clas12::dcana      analyzer;
clas12::sector     sector;

long static eventsProcessed = 0;

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
  reader.open(filename);
  reader.readDictionary(factory);
  //hitsBank = new hipo::bank(factory.getSchema("HitBasedTrkg::HBHits"));
  hitsBank = new hipo::bank(factory.getSchema("TimeBasedTrkg::TBHits"));
  trackBank = new hipo::bank(factory.getSchema("TimeBasedTrkg::TBTracks"));
  dc_tdc    = new hipo::bank(factory.getSchema("DC::tdc"));
  nnBank    = new hipo::bank(factory.getSchema("nn::dchits"),4);
  writer.addDictionary(factory);
  writer.open("ai_inferred.hipo");
}
//========================================================
//= Read next bunch of track candidates and return
//= the number of candidates that were read.
//= if end of file war reached, return value will be -1
//= otherwise number of bunches is returned (0 is also
//= valid return value).
//========================================================
int read_next(){
  bool status = reader.next();

  if(eventsProcessed>15000) status = false;
  if(status==false) {
    writer.close();
    return -1;
  }
  eventsProcessed++;
  if(eventsProcessed%1000==0){
    printf("processed %lu\n",eventsProcessed);
  }
  reader.read(event);
//  event.getStructure(*hitsBank);
//  event.getStructure(*trackBank);
  event.getStructure(*dc_tdc);

  sector.reset();
  //sector.read(*hitsBank,1);
  //sector.readTrackInfo(*trackBank);
  sector.readRaw(*dc_tdc,1);
  sector.makeTracks();
  //analyzer.readClusters(*hitsBank,1);
  //analyzer.makeTracks();
  //int ntracks = analyzer.getNtracks();
  return 1;
}

int count_roads(int banch){
  //int ntracks = analyzer.getNtracks();
  std::vector<double> features = sector.getFeatures();
  return features.size()/6;
}

void  read_roads ( double *roads, int nroads, int banch){

  std::vector<double> features = sector.getFeatures();
  for(int i = 0; i < features.size(); i++){
    roads[i] = features[i];
  }
  //memcpy(roads,&features[0],nroads*4*6);
  //analyzer.getFeatures(roads);
  /*int nfeatures = 6;
  for(int i = 0; i < nroads*nfeatures; i++){
    roads[i] = ((double)rand())/RAND_MAX;
  }*/
}

void  write_roads( double *roads_results, int nroads, int banch){

  printf("############ WRITE EVENT ##################\n");
  sector.setWeights(roads_results);
  sector.analyze();
  sector.showTrackInfo();
  //if(sector.getTrackCount()>2)
    sector.show();
    //sector.createWireHits(*nnBank);

      //sector.showBest();

      hipo::bank bank(factory.getSchema("nn::dchits"),5);
      sector.createWireHits(bank);

      if(bank.getRows()>0){
        event.addStructure(bank);
      }
      
      writer.addEvent(event);

      //bank.setRows(12);
      //bank.show();
      //nnBank->setRows(5);
      //nnBank->show();
      //if(bank.getRows()>5)
      /*for(int i = 0; i < bank.getRows(); i++){
        bank.putShort("index",i,(int16_t) 3);
        bank.putByte("id",i, (int8_t) 3);
        //int id = nnBank->getInt(0,i);
        //int index = nnBank->getInt(1,i);
        //printf("%5d : %5d\n",i, id);
        //printf("%5d : %5d %5d\n",i,id,index);
      }*/
      /*if(bank.getRows()>50)
      bank.show();
*/
  /*printf("banch %5d : results = ",banch);
  for(int i = 0; i < nroads; i++){
    printf(" %6.3f ",roads_results[i]);
  }
  printf("\n");*/
  /*
  analyzer.showFeatures();
  printf(" Probability : ");
  for(int i = 0; i < nroads; i++){
    printf("%8.4f ",roads_results[i]);
  }
  double  max = 0.0;
  double summ = 0.0;
  int    max_index = -1;

  for(int i = 0; i < nroads; i++){
    double value = roads_results[i];
    if(value>max){
      max_index = i;
      max = value;
    }
    summ += value;
  }*/

/*
  int counter = 0;
  double average = summ/nroads;
  for(int i = 0; i < nroads; i++){
    double value = roads_results[i];
    if(value>1.1*average) counter++;
  }
  printf("\n Max = %8.5f, Average = %8.5f, Counter = %8d, Index = %8d\n",max,summ/nroads,counter,max_index);
  if(max>0.75&&max_index>=0){
    std::vector<int> index = analyzer.getTrackIndex(max_index);
    if(index.size()<100){
      hipo::bank nnBank (factory.getSchema("nn::dchits"),index.size());
      for(int i = 0; i < index.size(); i++){
        nnBank.putByte(    "id", i, 1);
        nnBank.putInt ( "index", i, index[i]);
      }
      nnBank.show();
      event.addStructure(nnBank);
      writer.addEvent(event);
    }
  }*/
}
