#include "cfoam.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>

void foam_debug();
void readVtk(char filename[128], cnf::foam &f);


int main(int argc, char** argv){
  //char filename[128];
  char inputFile[256];
  printf("welcome to foam simulator.....\n");
  if(argc>1) {
    sprintf(inputFile,"%s",argv[1]);
    //sprintf(outputFile,"%s",argv[2]);
  } else {
    std::cout << " *** please provide a file name..." << std::endl;
    exit(0);
  }

  cnf::foam fm;
  readVtk(inputFile,fm);
  fm.init();
  //fm.show();
  std::vector<double> result;
  for(int i = 0; i < 10000; i++){
    fm.getRandom(result);
    printf("%12.4f %12.4f\n",result[0],result[1]);
  }
  //foam_debug();

  return 1;
}
//********************************************************
// READING VTK file into FOAM structure
//********************************************************
void readVtk(char *filename, cnf::foam &f){
  std::ifstream file(filename);
  size_t npoints, ncells;
      // skip first four lines
      std::string line,tmp;
      for(int i = 0 ; i < 4; i++)
      {
          std::getline(file,line);
      }
      file >> line ; // POINTS;
      file >> npoints;
      file >> line; // point type
      double px,py,pz;
      for(size_t i = 0 ; i < npoints; i++)
      {
        file >> px;//points[3*i];
        file >> py;//points[3*i + 1];
        file >> pz;//points[3*i + 2];
        f.addPoint(px,py);
      }
      std::cout << npoints <<"\n";
      while(line.find("CELLS") == std::string::npos)
      {
        std::getline(file,line);
      }
      std::stringstream ss(line);

      std::getline(ss,tmp,' '); // skip CELLS

      ss >> ncells;
      const int cell_npoints = 3; // 4 for 3D tetrahedral mesh
    std::vector<size_t> cells( cell_npoints * ncells);
    int ci1,ci2,ci3;
    for(size_t i = 0 ; i < ncells; i++)
    {
        size_t points_per_cell;

        file >> points_per_cell;
        assert(points_per_cell == cell_npoints);
        file >> ci1;//cells[cell_npoints*i ];
        file >> ci2;//cells[cell_npoints*i + 1];
        file >> ci3;//cells[cell_npoints*i + 2];
        cnf::cell c;
        c.addNode(ci1);
        c.addNode(ci2);
        c.addNode(ci3);
        f.addCell(c);
        //file >> cells[cell_npoints*i + 3]; for 3D
    }
    std::cout << ncells <<"\n";
   std::cout << cells[0] << " " << cells[1] << " " << cells[2] <<"\n";
   //std::cout << cells[3*(ncells - 1) ] << " " << cells[3*(ncells - 1) +1 ] << " " << cells[3*(ncells -1) + 2] <<"\n";

   // Read weights

   // Read until you find the POINTDATA  line
   while(line.find("POINT_DATA") == std::string::npos)
   {
       std::getline(file,line);
   }

   std::stringstream ss1(line);

   size_t nweights;
   std::getline(ss1,tmp,' '); // skip POINT_DATA

   ss1 >> nweights;
   assert(nweights == npoints);
    // skip next two lines
    std::getline(file,line);
    std::getline(file,line);

    std::vector<float> weights(npoints);

    for(size_t i = 0 ; i < npoints; i++)
    {
        file >> weights[i];
        f.setCellWeight(i,weights[i]);
    }
    std::cout << weights[0] << "\n";
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
