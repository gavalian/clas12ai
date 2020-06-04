#include <stdlib.h>
#include <stdio.h>

// Opens the file to be read.
void open_file(const char *filename){
    printf("---> some file was opened with name : %s\n",filename);
}

// Read the next 24 points into the roads array. These points will be used
// to predict the next points of the track.
// Returns 1 if points were written in roads, 0 if there are no points to be read.
int read_next(double* roads)
{

    static int read_once = 0;
    if ( read_once == 0)
    {
        read_once = 1;
        roads[0] = 29.00000;
        roads[1] = 30.00000;
        roads[2] = 29.00000;
        roads[3] = 29.00000;
        roads[4] = 29.00000;
        roads[5] = 0.00000;
        roads[6] = 30.00000;
        roads[7] = 30.00000;
        roads[8] = 29.50000;
        roads[9] = 30.00000;
        roads[10] = 29.00000;
        roads[11] = 30.00000;
        roads[12] = 25.00000;
        roads[13] = 26.00000;
        roads[14] = 25.00000;
        roads[15] = 25.00000;
        roads[16] = 24.00000;
        roads[17] = 0.00000;
        roads[18] = 24.00000;
        roads[19] = 24.00000;
        roads[20] = 24.00000;
        roads[21] = 24.00000;
        roads[22] = 23.00000;
        roads[23] = 23.00000;

        return 1;
    }

    return 0;
}


// Provides the points predicted in the roads array.
void write_roads(double* roads)
{
    for(int i=0; i< 24; i++)
    {
        printf("%lf,", roads[i]);
    }
    printf("\n");
}


