import json
import numpy as np
import time
import matplotlib.pyplot as plt
# import random
import sys
import pandas as pd
from joblib import Parallel, delayed
# from sklearn.datasets import dump_svmlight_file


def get_planes_rings_pads(df):
    """Given a dataframe containing tracking events translates hit ids
    to the correct coordinates

    Args:
        df (pandas.DataFrame): DataFrame containing tracking events 
        using ids for hitpoints

    Returns:
        numpy array: A numpy array containing the indexes in [eventid, x, y, z]
        format for the hitpoints detected
    """

    eventids = np.array(df['eventid']).reshape(-1, 1)
    ids = np.array(df['id']).reshape(-1, 1)
    planes = np.right_shift(ids, 18)
    rings = np.bitwise_and(np.right_shift(ids, 9), 0b0111111111)
    pads = np.bitwise_and(ids, 0b0111111111)

    return np.concatenate((eventids, planes, rings, pads), axis=1)

def compute_offset_from_center(values, ranges):
    """
    Given a 1D array of values and an array of ranges, determines the index of the range the values
    belong to, and then subtracts the center indexes.

    Args:
        values: Values to check
        ranges: Array containing possible ranges; must have an odd size

    Returns:
        Indexes offset of the values range from the range in the center indexes
    """

    value_range_indexes = np.empty(values.shape[0], dtype=np.int) 
    value_range_indexes.fill(int(len(ranges) / 2))

    for index in range(0, value_range_indexes.shape[0]):
        for i in range(0, len(ranges)):
            if ranges[i][0] <= values[index] < ranges[i][1]:
                value_range_indexes[index] = i
                break

    return value_range_indexes - len(ranges) // 2

def load_events(i,e):
    """Loads an event to an event and creates the respective DataFrames

    Args:
        i (int): The ID of the event
        e (dict): Dictionary containing tracks and hits info of an event

    Returns:
        [type]: [description]
    """
    hits_dataframes = []
    tracks_dataframes = []

    for tid, t in enumerate(e['tracks']):
        df = pd.DataFrame(data=[t['track']])
        df.insert(0, 'trackid', tid)
        df.insert(0, 'eventid', i)
        tracks_dataframes.append(df)
        # for s in t['hitlist']:
        if not t['hitlist']:
            continue
        df = pd.DataFrame(data=t['hitlist'])
        df.insert(0, 'trackid', tid)
        df.insert(0, 'eventid', i)
        df['track_z']  = t['track']['z']
        hits_dataframes.append(df)
    
    hits_dataframe = pd.concat(hits_dataframes, ignore_index = True)
    tracks_dataframe = pd.concat(tracks_dataframes, ignore_index = True)

    return (hits_dataframe, tracks_dataframe)

def load_json_to_dataframe(inpath, num_workers = -1):
    """Converts a 3d tracks input from json to panda DataFrame

    Args:
        inpath (string): [description]
        num_workers (int, optional): Number of process to use. Default is the number of available cpus.

    Returns:
        tuple of pandas.DataFrame: The tuple contains one dataframe consisting
        of the information provided per track and one with information
        about the hits per track
    """
    print("Opening json file in "+inpath)
    start = time.time()
    with open(inpath) as f:
        data = f.read()
        js = json.loads(data)

    events = js['events']
    end = time.time()
    print("Done in "+str(end-start)+" sec")

    hits_dataframes = []
    tracks_dataframes = []

    print("Loading json file to pandas DataFrame...")
    start = time.time()
    res = Parallel(n_jobs=num_workers, verbose = 1)(delayed(load_events)(i,event) for i,event in enumerate(events))
    for h, t in res:
        hits_dataframes.append(h)
        tracks_dataframes.append(t)
    
    hits_dataframe = pd.concat(hits_dataframes, ignore_index = True)
    tracks_dataframe = pd.concat(tracks_dataframes, ignore_index = True)
    end = time.time()
    print("Load finished in "+str(end-start)+" sec")

    return (hits_dataframe, tracks_dataframe)

def hits_array_from_parquet(path_to_parquet_file, z_offset_ranges = None):
    """Given the path to a parquet file containing the hits information
    returns a numpy array that represents the hits and is optionally 
    quantized

    Args:
        path_to_parquet_file (string): Path to the parquet file containing the hits info
        z_offset_ranges (list, optional): A list of ranges for the quantization process. Must be of odd number. Defaults to None.

    Returns:
        numpy array: A numpy array representing the hits read from the parquet file
        ready to be used for training
    """
    hits = None
    num_base_planes = 10
    num_quantized_planes = 10

    if z_offset_ranges == None:
        print('Extracting hitpoints from ids...')
        start = time.time()
        df = pd.read_parquet(path_to_parquet_file, columns=['eventid', 'id'])

        hits = np.zeros((len(df.eventid.unique()), 10, 21, 122))

        opt_res = get_planes_rings_pads(df)
        end = time.time()
        print('Done in: '+str(end-start)+" sec")

    else:
        print('Extracting hitpoints from ids and quantizing...')
        start = time.time()
        num_planes_between = int(len(z_offset_ranges) / 2) # Number of added planes between the base planes
        num_quantized_planes = num_base_planes + (num_base_planes + 1) * num_planes_between # Number of base planes plus the number of added planes

        df = pd.read_parquet(path_to_parquet_file, columns=['eventid', 'id', 'track_z'])

        hits = np.zeros((len(df.eventid.unique()), num_quantized_planes, 21, 122))

        opt_res = get_planes_rings_pads(df)

        opt_res[:,1] = (opt_res[:,1] * num_planes_between) + num_planes_between
        opt_res[:,1] += compute_offset_from_center(np.array(df['track_z']), z_offset_ranges)
        end = time.time()
        print('Done in: '+str(end-start)+" sec")

        
    hits[opt_res[:,0], opt_res[:,1], opt_res[:,2], opt_res[:,3]] = 1
    
    return (hits, num_quantized_planes)

def coordinates_dataframe_from_parquet(path_to_parquet_file, z_offset_ranges = None):
    """Given the path to a parquet file containing the hits information
    returns a dataframe that represents the hits in (plane, ring, pad) coordinates and is optionally 
    quantized

    Args:
        path_to_parquet_file (string): Path to the parquet file containing the hits info
        z_offset_ranges (list, optional): A list of ranges for the quantization process. Must be of odd number. Defaults to None.

    Returns:
        dataframe: A dataframe representing the hits read from the parquet file
        in plane, ring, pad coordinates
    """

    if z_offset_ranges == None:
        print('Extracting hitpoints coordinates from ids...')
        start = time.time()
        df = pd.read_parquet(path_to_parquet_file, columns=['trackid', 'eventid', 'id'])

        opt_res = get_planes_rings_pads(df)
        end = time.time()
        print('Done in: '+str(end-start)+" sec")

    else:
        print('Extracting hitpoints coordinates from ids and quantizing...')
        start = time.time()
        num_planes_between = int(len(z_offset_ranges) / 2) # Number of added planes between the base planes

        df = pd.read_parquet(path_to_parquet_file, columns=['trackid', 'eventid', 'id', 'track_z'])

        opt_res = get_planes_rings_pads(df)

        opt_res[:,1] = (opt_res[:,1] * num_planes_between) + num_planes_between
        opt_res[:,1] += compute_offset_from_center(np.array(df['track_z']), z_offset_ranges)
        end = time.time()
        print('Done in: '+str(end-start)+" sec")

    df['hits_evetid'] = opt_res[:,0]  
    df['plane'] = opt_res[:,1]  
    df['ring'] = opt_res[:,2]  
    df['pad'] = opt_res[:,3]  
    
    return df

def save_to_parquet(hits_df, tracks_df, oupaths):
    """Saves pandas hits and tracks information dataframes to the disk in parquet format

    Args:
        hits_df (pandas.DataFrame): DataFrame containing information per hit
        tracks_df (pandas.DataFrame): DataFrame containing information per track
        oupaths (tuple of strings): Tuple of paths to store hits and tracks DataFrames in parquet format
    """
    start = time.time()
    hits_df.to_parquet(oupaths[0], index=False)
    tracks_df.to_parquet(oupaths[1], index=False)
    end = time.time()
    print("Stored to parquet files in " + str(end-start) +" sec" )

z_offset_ranges = [(-20, -15), (-15, -5), (-5, 5), (5, 15), (15, 21)]

# if len(sys.argv) != 2:
#     print("Please provide input")
#     exit()

#inpath = sys.argv[1]

#hits_df, tracks_df = load_json_to_dataframe(inpath)
#save_to_parquet(hits_df, tracks_df, ('hits_250.parquet', 'tracks_250.parquet'))
#hits, num_quantized_planes = hits_array_from_parquet('hitsblah.parquet', z_offset_ranges)
# plt.imsave('img_test.png', hits[0].reshape(num_quantized_planes * 21, 122))
# # np.save(outpath, hits)