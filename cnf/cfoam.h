/**
 * Event generator based on VTK file.
 * author: G.Gavalian (Dec 06 2019)
*/

#include <iostream>
#include <vector>

namespace cnf {

class points {
    int ndim;
    std::vector<double> buffer;
  public:
    points(){ ndim = 2;}
    ~points(){}

    void addPoint(double x, double y){
      buffer.push_back(x);
      buffer.push_back(y);
    }
};

class cell {

    std::vector<int> corners;
    double weight;

  public:
    cell(){}
    cell(const cell &c){ corners = c.corners; weight = c.weight;}
    ~cell(){}

    void    addNode(int nid){corners.push_back(nid);}
    void    setWeight(double w){weight = w;}
    double  getWeight(){return weight;}
    std::vector<int>    &getNodes(){ return corners;}
};

class foam {

  points               pointBuffer;
  std::vector<cell>    cellBuffer;
  std::vector<double>  cellIntegral;

  public:

    foam(){}
    ~foam(){}

    void addPoint(double x, double y){ pointBuffer.addPoint(x,y);}
    void addCell(cell &cell) { cellBuffer.push_back(cell);}
    void init();
    void show();
    void getRandom(std::vector<double> &values);
    int  getBin(double value);
};
}
