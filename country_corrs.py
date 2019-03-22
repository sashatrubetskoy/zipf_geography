import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.text import TextPath
from PIL import Image
from matplotlib import rcParams
from scipy.signal import savgol_filter
from scipy.stats.stats import pearsonr

def render_pop(arr):
    # arr = 1 - (arr/arr.max())
    plt.imshow(arr)
    plt.show()
    return None

def zipf_corr(pops):
    ranks = list(range(1, len(pops)+1))[::-1]
    return pearsonr(np.log10(ranks), np.log10(pops))[0]

df = pd.read_csv('gpw_v4_national_identifier_grid_rev11_lookup.txt', sep='\t')
conv = df[['Value', 'ISOCODE']].set_index('Value').to_dict()['ISOCODE']

im60 = Image.open('gpw_v4_population_count_rev10_2020_1_deg.tif')
imc60 = Image.open('gpw_v4_national_identifier_grid_rev11_1_deg.tif')
im30 = Image.open('gpw_v4_population_count_rev11_2020_30_min.tif')
imc30 = Image.open('gpw_v4_national_identifier_grid_rev11_30_min.tif')

ar60 = np.array(im60)[1:-4]
arc60 = np.array(imc60)[6:-34]
ar30 = np.array(im30)
arc30 = np.array(imc30)

ar60 = np.clip(ar60, a_min=0, a_max=ar60.max())
ar30 = np.clip(ar30, a_min=0, a_max=ar30.max())

result = {}
for country in np.unique(arc30):
    if country != 32767:
        country_subset = ar30[(arc30 == country)]
        pops = sorted(country_subset.flatten())
        pops = [p for p in pops if p >= 10000]
        if pops:
            corr = zipf_corr(pops)
            result[conv[country]] = corr
        else:
            result[conv[country]] = 0

with open('country_corrs.csv', 'w') as f:
    f.write('iso3,corr\n')
    for r in result:
        if not result[r] >= -1:
            result[r] = 0
        f.write('{},{}\n'.format(r, result[r]))