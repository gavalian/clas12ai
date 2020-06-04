import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer


class CLibInterface():

    def __init__(self,c_lib):
        """Initializes the python binding for the C library
        
        Parameters:
        c_lib: The C library to use. If the library is not in the LD_LIBRARY_PATH
               a full path must be provided.
        
        """

        ## Open the C library ##
        self.c_interface = ctypes.CDLL(c_lib)

        ## Create python bindings for the C functions   ##
        self.c_interface.open_file.argtypes = [ctypes.c_char_p]
        self.c_interface.open_file.restype = None

        self.c_interface.read_next.argtypes = None
        self.c_interface.read_next.restype  = ctypes.c_int

        self.c_interface.count_roads.argtypes  = [ctypes.c_int]
        self.c_interface.count_roads.restype  = ctypes.c_int

        self.c_interface.read_roads.argtypes = ctypes.c_void_p,ctypes.c_int,ctypes.c_int
        self.c_interface.read_roads.restype  = None

        self.c_interface.write_roads.argtypes = ctypes.c_void_p,ctypes.c_int,ctypes.c_int
        self.c_interface.read_roads.restype  = None
    
    def open_file(self,path_to_file):
        """Open the specified file through the C library

        Parameters:
        path_to_file: Path to the file to open from the C library

        """
        
        c_str_path = path_to_file.encode('utf-8')
        self.c_interface.open_file(c_str_path)

    def read_next(self):
        """Get the number of banches

        Returns: 
        int: The number of banches read from the file
        """

        return self.c_interface.read_next()

    def count_roads(self,banch):
        """Counts the number of roads in a banch
        
        Parameters:
        banch: The banch to count the roads for

        Returns:
        int: The numbe of roads in the specified banch
        """

        return self.c_interface.count_roads(banch)
    
    def read_roads(self,nroads,banch):
        """Reads nroads from the specified banch

        Parameters:
        nroads: The number of roads to read
        banch:  The banch to read the roads from

        Returns:
        A numpy array with the roads read for the specified banch
        """

        roads = np.zeros((nroads,6),dtype = float)
        self.c_interface.read_roads(roads.ctypes.data_as(ctypes.c_void_p),ctypes.c_int(nroads),ctypes.c_int(banch))

        return roads

    def write_roads(self,res_roads,banch):
        """Writes the predictions got from a set of roads to the specified banch
        
        Parameters:
        res_roads: A numpy array with the predicted probabilities for each road
        banch : The banch that those roads belong to
        """
        self.c_interface.write_roads(np.array(res_roads).ctypes.data_as(ctypes.c_void_p),ctypes.c_int(res_roads.shape[0]),ctypes.c_int(banch))