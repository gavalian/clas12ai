import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# 21 planes
#z_offset_ranges = [(-20, -1.5), (-1.5, 1.5), (1.5, 21)]

# 32 planes
z_offset_ranges = [(-20, -10), (-10, -1.5), (-1.5, 1.5), (1.5, 10), (10, 21)]

# 43 planes
#z_offset_ranges = [(-20, -15), (-15, -10), (-10, -1.5), (-1.5, 1.5), (1.5, 10), (10, 15), (15, 21)]

NUM_PLANES = 10 # Base number of planes
NUM_PLANES_BETWEEN = int(len(z_offset_ranges) / 2) # Number of added planes between the base planes
NUM_QUANTIZED_PLANES = NUM_PLANES + (NUM_PLANES + 1) * NUM_PLANES_BETWEEN # Number of base planes plus the number of added planes

def get_plane_ring_pad(id):
    plane = id >> 18
    ring = (id >> 9) & 0b0111111111
    pad = id & 0b0111111111

    if plane >= 10 or ring >= 21 or pad >= 122:
        print("ERROR COORDINATE")
        exit()

    return (plane, ring, pad)


def compute_offset_from_center(value, ranges):
    """
    Given a value and an array of ranges, determines the index of the range the value
    belongs to, and then subtracts the center index.

    Args:
        value: Value to check
        ranges: Array containing possible ranges; must have an odd size

    Returns:
        Index offset of the value range from the range in the center index
    """

    center_index = int(len(ranges) / 2)

    value_range_index = center_index
    for i in range(0, len(ranges)):
        if ranges[i][0] <= value < ranges[i][1]:
            value_range_index = i
            break

    return value_range_index - center_index


if len(sys.argv) != 3:
    print("Please provide input and output path")
    exit()

inpath = sys.argv[1]
outpath = sys.argv[2]

with open(inpath) as f:
    data = f.read()
    js = json.loads(data)

events = js['events']

global_hitlist = []
track_zs = []
hitlist_zs = []
for i, event in enumerate(events):
    for t in event['tracks']:
        track_zs.append(t['track']['z'])
        t_hitlist = []

        for s in t['hitlist']:
            hitlist_zs.append(s['ztrue'])
            plane, ring, pad = get_plane_ring_pad(s['id'])

            # Compute quantized z offset
            plane += (plane * NUM_PLANES_BETWEEN) + NUM_PLANES_BETWEEN
            plane += compute_offset_from_center(t['track']['z'], z_offset_ranges)

            global_hitlist.append((i,) + (plane, ring, pad))

hits = np.zeros((len(events), NUM_QUANTIZED_PLANES, 21, 122))
e, x, y, z = zip(*global_hitlist)
hits[e, x, y, z] = 1

plt.imsave('quantization.png', hits[0].reshape(NUM_QUANTIZED_PLANES * 21, 122))
np.save(outpath, hits)
