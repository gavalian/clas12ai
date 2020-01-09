#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>

const char* filename = "dc_wire_map.txt";


class Point
{

public:
    Point(int x, int y)
    {
        this->x = x;
        this->y = y;
        label = -2;
    }

    friend std::ostream& operator<<(std::ostream &out, const Point &p) 
    {
        return out << "(" << p.x << "," << p.y <<","<< p.label <<")";
    }


    int x,y,label;
};

bool comp (const Point* p1, const Point* p2)
{
    if (p1->label == p2->label)
    {
        return p1->y>p2->y;
    }
    return p1->label>p2->label;
}

int distance(Point* p1, Point* p2)
{
    int dist = (p1->x - p2->x);
    if ( dist < 0)
    {
        return -dist;
    } 

    return dist;
}

void read_next_dataset(FILE* fp, std::vector<Point*>* points)
{
    for (int i = 5; i >= 0; i--)
    {
        for(int j=0; j < 112; j++)
        {
            int temp = fgetc(fp);
            if ( temp == EOF )
            {
                return ;
            }
            if ( temp == '\n' )
            {
                temp = fgetc(fp);
            }

            if ( temp == 'X')
            {
                points->push_back(new Point(j,i));
            }

            
        }
    }
    fgetc(fp);
}

std::vector<Point*> range_query(std::vector<Point*>* points, Point* s, int eps)
{
    std::vector<Point*> neighbors;
    int min = 10000;
    int start = s->y-1;
    int i = start;
    Point* q = s;
    while (i>=0)
    {
        Point* point = NULL;
        min = 10000;
        for (size_t j=0; j< points[i].size() ; j++)
        {
            Point* p = points[i][j];
            int dist = distance(q,p);

            if (dist <= eps * (start-i+1))
            {
                if ( dist < min )
                {
                    if ( p->label == -2)
                    {
                        point = p;
                        min = dist;
                    }
                }

            }
        }

        if (point != NULL)
        {
            neighbors.push_back(point);
            q = point;
            start = i-1;

        }
        i -= 1;
    }
    
    return neighbors;
}

/** Given a set of points representing detections in the drift chambers, assigns each point to a cluster
 *  Points assigned label -1 are tagged as noise.
 * 
 *  This algorithm works similarly to the DBSCAN algorithm, specifically:
 *  Starting from the top layer once a point is found we try we find the closest point in the
 *  next layer up to eps distance in the x-axis. Next we start from this point and find the closest
 *  one in the next layer. If a point is not found in distance eps we check the following layer but
 *  the maximum distance is now doubled.
 *  This continues until we reach the last layer. If the number of points found when reaching the
 *  last layer is at least minPts a cluster is formed.
 * 
 *  @param points A vector containing the points of detections in the drift chambers
 *  @param eps The maximum distance in the x-axis to look for points to form a cluster
 *  @param minPts The minimum number of points that can form a cluster  
 */
void find_clusters(std::vector<Point*>& points, int eps, size_t minPts){
    int C = -1;
    std::vector<Point*> all_points[6];
    for (size_t i = 0; i< points.size(); i++)
    {
        all_points[points[i]->y].push_back(points[i]);
    }
    int i = 5;
    while (i >= 0) 
    {
        for (size_t j = 0; j< all_points[i].size(); j++)
        {
            Point* p = all_points[i][j];
            if (p->label != -2)
            {
                continue;
            }

            std::vector<Point*> neighbors = range_query(all_points, p, eps);
        
            if ( neighbors.size() < (minPts-1) )
            {
                p->label = -1;
                continue;
            }

            C += 1;
            p->label = C;

            for(size_t k = 0; k< neighbors.size(); k++)
            {
                Point* q = neighbors[k];
                q->label = C;
            }
        } 

        bool increase_it = true;
        std::vector<Point*>::iterator it = all_points[i].begin();
        while ( it != all_points[i].end())
        {
            increase_it = true;
            Point* p = *it;
            if (p->label == -1)
            {
                std::vector<Point*>::iterator it2 = all_points[i].begin();
                while ( it2 != all_points[i].end())
                {
                    Point* q = *it2;
                    if ( (distance(p,q) == 1) && q->label != -1)
                    {
                        p->label = q->label;
                        if( it2 < it)
                        {
                            it--;
                        }
                        it2 = all_points[i].erase(it2);
                        it = all_points[i].erase(it);
                        increase_it = false;
                        break;
                    }
                    else
                    {
                        ++it2;
                    }
                    
                }
            }
            if ( increase_it)
            {
                it++;
            }
        }
        i -= 1;
    }
}

int dataset_case = 6;

int main(int argc, char** argv)
{
    FILE* fp = fopen(filename,"r");
    std::vector<Point*> points;
    int curr_case = 0;
    // int case_ = atoi(argv[1]);
    // dataset_case = case_;
    read_next_dataset(fp, &points);

    while (curr_case != dataset_case && !points.empty())
    {
        for(size_t i = 0; i< points.size(); i++)
        {
            delete points[i];
        }
        points.clear();
        read_next_dataset(fp, &points);
        curr_case ++;
    }

    find_clusters(points, 2, 4);

    std::sort(points.begin(), points.end(), comp);
    for ( std::vector<Point*>::iterator it = points.begin(), end = points.end(); it != end ; ++it)
    {
        std::cout<< (**it);
        if ( (it+1) != end )
        {
            std::cout<< ", ";
        } 
    }

    std::cout<<std::endl;
    fclose(fp);

    for(size_t i = 0; i< points.size(); i++)
    {
        delete points[i];
    }
    points.clear();

    return 0;
}
