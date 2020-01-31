import ctypes
import numpy as np


class CLibInterface:
    """
    Provides an interface to the native shared object library
    that is used for reading input data and writing predictions.
    """

    def __init__(self, c_lib_path):
        """
        Initializes the python binding for the C library

        Args:
            c_lib_path: The C library to use. If the library is not in the LD_LIBRARY_PATH
                a full path must be provided.
        """

        # Open the C library
        self.c_interface = ctypes.CDLL(c_lib_path)

        # Create python bindings for the C functions
        self.c_interface.open_file.argtype = ctypes.c_char_p
        self.c_interface.open_file.restype = None

        self.c_interface.read_next.argtype = ctypes.c_void_p
        self.c_interface.read_next.restype = ctypes.c_int

        self.c_interface.write_roads.argtype = ctypes.c_void_p
        self.c_interface.write_roads.restype = None

    def open_file(self, path_to_file):
        """
        Open the specified file through the C library

        Args:
            path_to_file: Path to the file to open from the C library
        """

        c_str_path = path_to_file.encode('utf-8')
        self.c_interface.open_file(c_str_path)

    def read_next(self):
        """
        Get the next incomplete track

        Returns:
            numpy array: The observations read from the file
        """

        observations = np.ones((1, 24), dtype=np.double)
        found = self.c_interface.read_next(observations.ctypes.data_as(ctypes.c_void_p))
        
        if found == 0 :
            observations = np.zeros(1)

        return observations

    def write_roads(self, res_roads):
        """
        Writes the predictions that were made based on the incomplete track

        Args:
            res_roads: A numpy array with the predicted track
        """

        self.c_interface.write_roads(np.array(res_roads,dtype=np.double).ctypes.data_as(ctypes.c_void_p))
