#===============================================================
# Makefile for compilation of C_PYTHON interface for AI project
# author: G.Gavalian (Nov. 20 2019)
#===============================================================

SOURCES := $(wildcard r*.c)
OBJECTS := $(SOURCES:.c=.o)

INCLUDES=-I$(HIPO_LIB)/hipo4 -I$(HIPO_LIB)/ai
LIBRARIES=-L$(HIPO_LIB)/lib -llz4 -lhipo4 -ltrack

CXXFLAGS = -std=c++11

lib: $(OBJECTS)
	$(CXX) -shared $(OBJECTS) $(LIBRARIES) -fPIC -o libread.so

clean:
	@echo removing object files
	@rm -rf *.o lib*so

%.o: %.c
	$(CXX) -fPIC -c $< -O2 -D__LZ4__ $(CXXFLAGS) $(INCLUDES)
