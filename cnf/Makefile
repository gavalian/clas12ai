#===============================================================
# Makefile for compilation of C_PYTHON interface for AI project
# author: G.Gavalian (Nov. 20 2019)
#===============================================================

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.c=.o)

INCLUDES=-I.
#LIBRARIES=-L$(HIPO_LIB)/lib -llz4 -lhipo4 -ltrack

CXXFLAGS = -std=c++11


all: lib exe
	@echo ' '
	@echo '>>>> distribution build success...'
	@echo ' '

lib: $(OBJECTS)
	$(CXX) -shared $(OBJECTS) $(LIBRARIES) -fPIC -o libfoam.a

exe: main.o
	$(CXX) -o main.exe main.o libfoam.a

clean:
	@echo removing object files
	@rm -rf *.o lib*a *.exe

%.o: %.cpp
	$(CXX) -fPIC -c $< -O2 -D__LZ4__ $(CXXFLAGS) $(INCLUDES)

%.o: %.ccc
	$(CXX) -fPIC -c $< -O2 -D__LZ4__ $(CXXFLAGS) $(INCLUDES)
