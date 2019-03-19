import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image
from scipy.stats.stats import pearsonr


def render_pop(arr):
    # arr = 1 - (arr/arr.max())
    plt.imshow(arr)
    plt.show()
    return None
    
# Moscow is at (lat, lon) = 55 N, 37 E; which translates to (i, j) = 29, 217
def get_ij(lat, lon, resolution):
    if resolution == 60:
        lat = int(lat)
        lon = int(lon)
        i = 84 - lat
        j = 180 + lon
    elif resolution == 30:
        lat = int(2*lat)/2
        lon = int(2*lon)/2
        i = 2*(89.5 - lat)
        j = 2*(180 + lon)
    return (int(i), int(j))


def zipf_graph(chunk):
    pops = sorted(chunk.flatten())
    ranks = list(range(1, len(pops)+1))[::-1]
    plt.scatter(np.log10(ranks), np.log10(pops))
    plt.show()

def get_pops(arr, centers):
    R = D // 2
    pops = []
    for c in centers:
        i0, j0 = c
        chunk = arr[i0-R:i0+R+1, j0-R:j0+R+1]
        pops = pops + list(chunk.flatten())
        pops = [p if p != 0 else 1 for p in pops] # Remove zeros for log
    return sorted(pops)

def zipf_corr(pops):
    ranks = list(range(1, len(pops)+1))[::-1]
    return pearsonr(np.log10(ranks), np.log10(pops))[0]

def get_candidate_centers(arr, chunk_centers):
    candidate_centers = set()
    for cc in chunk_centers:
        candidate_centers.add((cc[0], cc[1]+D))
        candidate_centers.add((cc[0], cc[1]-D))
        candidate_centers.add((cc[0]+D, cc[1]))
        candidate_centers.add((cc[0]-D, cc[1]))
    candidate_centers = [c for c in candidate_centers if c not in chunk_centers]
    candidate_centers = [c for c in candidate_centers if arr[c] != 0]
    return candidate_centers

def get_best_adjacent_center(arr, chunk_centers, pops, top_k=1):
    candidate_centers = get_candidate_centers(arr, chunk_centers)

    cands_and_corrs = []
    for c in candidate_centers:
        pops_c = sorted(pops + get_pops(arr, [c]))
        corr = zipf_corr(pops_c)
        cands_and_corrs.append((corr, c))

    if top_k < 1:
        how_many = int(len(cands_and_corrs) * top_k)
        if how_many < 1:
            how_many = 1
        top_k_candidates = [tup[1] for tup in sorted(cands_and_corrs)[:how_many]]
    else:
        top_k_candidates = [tup[1] for tup in sorted(cands_and_corrs)[:top_k]]
    new_pops = sorted(pops + get_pops(arr, top_k_candidates))
    new_corr = zipf_corr(new_pops)
    return top_k_candidates, new_corr, new_pops

def map_centers(arr, centers):
    R = D // 2
    new_arr = arr.copy()
    for i, c in enumerate(centers):
        i0, j0 = c
        new_arr[i0-R:i0+R+1, j0-R:j0+R+1] += (i+1)*5e5
    plt.imshow(new_arr)
    plt.show()
    return new_arr

def centers_to_latlon(centers, counts, resolution=30):
    result = []
    for k, c in enumerate(centers):
        i, j = c
        if resolution == 60:
            lat = 84 - i
            lon = j - 180
        elif resolution == 30:
            lat = 89.5 - i/2
            lon = j/2 - 180
        result.append([lat, lon, k, 0])
    result = np.array(result)
    for count in counts:
        result[count:, 3] += 1
    return result


im = Image.open('gpw_v4_population_count_rev10_2020_1_deg.tif')
arr = np.array(im)
arr = np.clip(arr, a_min=0, a_max=arr.max())

D = 1
# chunk_centers = [get_ij(40, 116)]
# seed = get_ij(55.8, 37.6, resolution=60) # Moscow
# seed = get_ij(48.8567, 2.3508, resolution=60) # Paris
seed = get_ij(40.7127, -74.0059, resolution=60) # NYC
chunk_centers = [(seed[0]+a, seed[1]+b) for a, b in zip([1,1,0,0],[1,0,1,0])] if D == 1 else [seed]
pops = [arr[c] if arr[c]!=0 else 1 for c in chunk_centers]
corrs = []
counts = []
for i in range(150):
    print('Iteration {}...'.format(i))
    most_recent_corr = corrs[-1] if corrs else -1
    prop = -most_recent_corr - 0.65
    print('\tprop={}'.format(prop))
    best_adjacent_centers, new_corr, new_pops = get_best_adjacent_center(arr, chunk_centers, pops, top_k=prop)
    
    corrs.append(new_corr)
    chunk_centers.extend(best_adjacent_centers)
    pops = new_pops
    counts.append(len(chunk_centers))

N = 7
smooth = savgol_filter(corrs, N, 2)
fig, ax = plt.subplots()
ax.plot(corrs)
ax.plot(smooth)
plt.savefig('curve.png')
plt.close()

# min_smooth = list(smooth).index(min(smooth))
# map_centers(arr, chunk_centers[:min_smooth])
# map_centers(arr, chunk_centers)

latlon = centers_to_latlon(chunk_centers, counts, resolution=60)
with open('centers.csv', 'w') as f:
    f.write('Y,X,order,iteration\n')
    for l in latlon:
        f.write(','.join([str(e) for e in l])+'\n')