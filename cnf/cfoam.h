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
    std::vector<double> weights;

  public:
    points(){ ndim = 2;}
    ~points(){}

    void addPoint(double x, double y){
      buffer.push_back(x);
      buffer.push_back(y);
    }
    void    initWeights();
    void    setWeight(int index, double w);
    double  getWeight(std::vector<int> ind);

    int     getSize();
    double  getPoint(int index, int component);
    void    getRandom(std::vector<int> &index, std::vector<double> &result);
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
    void setCellWeight(int index, double weight) { cellBuffer[index].setWeight(weight);}

    void initPointWeights();
    void setPointWeight(int index, double w);

    void init();
    void show();
    void getRandom(std::vector<double> &values);
    int  getBin(double value);
};
}
