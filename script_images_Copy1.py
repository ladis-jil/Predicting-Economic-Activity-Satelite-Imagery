import pandas as pd
import numpy as np
import math
import rasterio
import random
import random
from sklearn.mixture import GaussianMixture as GMM
from utils import PlanetDownloader
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import h5py

load_dotenv()
PLANET_API_KEY= os.getenv('PLANET_API_KEY')

RANDOM_SEED = 7


geo_data      = pd.read_csv(f'raw_data/geo_eth.csv', sep=',')
cons_data     = pd.read_csv(f'raw_data/cons_eth.csv', sep=',')
geo_nig_data  = pd.read_csv(f'raw_data/geo_nig.csv', sep=',')
cons_nig_data = pd.read_csv(f'raw_data/cons_nig.csv', sep=',')

def process_ethiopia():

    consumption_pc_col = 'total_cons_ann' # per capita
    hhsize_col = 'hh_size' # people in household

    lat_col = 'lat_dd_mod'
    lon_col = 'lon_dd_mod'

    # purchasing power parity for ethiopia in 2015 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=ET)
    ppp = 7.882

    # for file in [consumption_file, geovariables_file]:
    #     assert os.path.isfile(os.path.join(lsms_dir, file)), print(f'Could not find {file}')

    df = cons_data
    df['cons_ph'] = df[consumption_pc_col] * df[hhsize_col]
    df['pph'] = df[hhsize_col]
    df['cons_ph'] = df['cons_ph'] / ppp / 365
    df = df[['household_id2', 'cons_ph', 'pph']]

    df_geo = geo_data
    df_cords = df_geo[['household_id2', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df, df_cords, on='household_id2')
    df_combined.drop(['household_id2'], axis=1, inplace=True)
    df_combined.dropna(inplace=True) # can't use na values

    df_clusters = df_combined.groupby(['cluster_lat', 'cluster_lon']).sum().reset_index()
    df_clusters['cons_pc'] = df_clusters['cons_ph'] / df_clusters['pph'] # divides total cluster income by people
    df_clusters['country'] = 'eth'
    return df_clusters[['country', 'cluster_lat', 'cluster_lon', 'cons_pc']]

def process_nigeria():

    consumption_pc_col = 'totcons' # per capita
    hhsize_col = 'hhsize' # people in household
    lat_col = 'LAT_DD_MOD'
    lon_col = 'LON_DD_MOD'

    # purchasing power parity for nigeria in 2015 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=NG)
    ppp = 95.255

    df = cons_nig_data
    df['cons_ph'] = df[consumption_pc_col] * df[hhsize_col]
    df['pph'] = df[hhsize_col]
    df['cons_ph'] = df['cons_ph'] / ppp / 365
    df = df[['hhid', 'cons_ph', 'pph']]

    df_geo = geo_nig_data
    df_cords = df_geo[['hhid', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df, df_cords, on='hhid')
    df_combined.drop(['hhid'], axis=1, inplace=True)
    df_combined.dropna(inplace=True) # can't use na values

    df_clusters = df_combined.groupby(['cluster_lat', 'cluster_lon']).sum().reset_index()
    df_clusters['cons_pc'] = df_clusters['cons_ph'] / df_clusters['pph'] # divides total cluster income by people
    df_clusters['country'] = 'ng'
    return df_clusters[['country', 'cluster_lat', 'cluster_lon', 'cons_pc']]


def main():
    data_eth = process_ethiopia()

    data_nig = process_nigeria()

    transform, tif_array = open_tif_image()

    add_nightlights(data_eth, tif_array, transform)

    add_nightlights(data_nig, tif_array, transform)

    df_eth_download = generate_download_locations(data_eth)

    df_nig_download =  generate_download_locations(data_nig)

    df_nig_download["country"] = "nig"
    df_eth_download["country"] = "eth"
    df_potential_download = pd.concat([ df_eth_download, df_nig_download], axis=0)
    df_potential_download.reset_index(drop=True, inplace=True)

    df_mod_download = drop_0s(df_potential_download, fr=0.1)

    df_mod_download = drop_in_range(df_mod_download, lower=0.001, upper=3, fr=0.4)

    df_mod_download = drop_0s(df_mod_download, fr=0.2)

    label0_max = 0.05
    label1_max = 5
    label2_max = 70

    df_download = df_mod_download.copy()

    create_nightlights_bin(df_download, cutoffs=[label0_max, label1_max, label2_max])
    index_list = []
    path='/home/jupyter/Predicting-Economic-Activity-Satelite-Imagery/image_download'
    for filename in os.listdir(path):
        if filename.endswith('.npz'):
            file_path = os.path.join(path, filename)
            index = int(filename.split('_')[-1].split('.')[0])
            index_list.append(index)
    df_download.iloc[index_list].to_csv('download_preprocess/img_info.csv')

    # client = PlanetDownloader(PLANET_API_KEY)
    
#     with h5py.File('image_matrix.hdf5', 'w') as f:
#         # Inicializar el dataset dentro del archivo HDF5
#         # Asumimos un máximo estimado de imágenes; ajusta según sea necesario
#         max_images = len(df_download)
#         images_dset = f.create_dataset('images', (max_images, img_height, img_width, channels), dtype='uint8')
#         error_index = []

#         print(f"Ready for download {max_images} images")
#         count = 0  # Contador para saber cuántas imágenes válidas hemos guardado

#         for i, row in df_download.iterrows():
#             lat = row['image_lat']
#             lon = row['image_lon']
#             try:
#                 img = client.download_image(lat, lon, 2015, 1, 2016, 12)
#                 if img is not None:
#                     # Asumimos que img ya está correctamente formateada como un numpy array de dtype 'uint8'
#                     images_dset[count] = img[..., :3]  # Guardar la imagen en el dataset
#                     count += 1
#                     np.savez(f"image_download/image_{i}", *np.array(img[..., :3]))
#                     print(f"{str(i+1)}/{str(max_images)}")
#             except Exception as e:
#                 error_index.append(i)
#                 print(f"error at index {i}, pass: {str(e)}")

#         # Si no todas las imágenes son válidas, puedes redimensionar el dataset para que solo tenga las imágenes válidas
#         if count < max_images:
#             images_dset.resize((count, img_height, img_width, channels))

#         # Guardar índices de error si hay algunos
#         if error_index:
#             error_np = np.array(error_index)
#             np.save('error_indexes.npy', error_np)

        # print(f"Successfully downloaded and saved {count} images.")

def create_nightlights_bin(df, cutoffs):
    assert len(cutoffs) >= 2, print('need at least 2 bins')
    cutoffs = sorted(cutoffs, reverse=True)
    labels = list(range(len(cutoffs)))[::-1]
    df['nightlights_bin'] = len(cutoffs)
    for cutoff, label in zip(cutoffs, labels):
        df['nightlights_bin'].loc[df['nightlights'] <= cutoff] = label


def create_space(lat, lon, s=10):
    """Creates a s km x s km square centered on (lat, lon)"""
    v = (180/math.pi)*(500/6378137)*s # roughly 0.045 for s=10
    return lat - v, lon - v, lat + v, lon + v

def open_tif_image():
    url_image = "raw_data/picture.tif"
    with rasterio.open(url_image) as src:
        image_data = src.read(1)
        transform = src.transform
        tif_array = np.squeeze(image_data)
        return transform, tif_array


def custom_rasterio_open(max_lon, max_lat, transform):
    xmaxPixel, yminPixel = ~transform * (max_lon, max_lat)
    xmaxPixel, yminPixel = int(xmaxPixel), int(yminPixel)
    return xmaxPixel, yminPixel


def add_nightlights(df, tif_array, transform):
    '''
    This takes a dataframe with columns cluster_lat, cluster_lon and finds the average
    nightlights in 2015 using a 10kmx10km box around the point

    I try all the nighlights tifs until a match is found, or none are left upon which an error is raised
    '''
    cluster_nightlights = []
    for i,r in df.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)

        xminPixel, ymaxPixel = custom_rasterio_open(min_lon, min_lat, transform)
        xmaxPixel, yminPixel = custom_rasterio_open(max_lon, max_lat, transform)
        assert xminPixel < xmaxPixel, print(r.cluster_lat, r.cluster_lon)
        assert yminPixel < ymaxPixel, print(r.cluster_lat, r.cluster_lon)
        if xminPixel < 0 or xmaxPixel >= tif_array.shape[1]:
            print(f"no match for {r.cluster_lat}, {r.cluster_lon}")
            raise ValueError()
        elif yminPixel < 0 or ymaxPixel >= tif_array.shape[0]:
            print(f"no match for {r.cluster_lat}, {r.cluster_lon}")
            raise ValueError()
        xminPixel, yminPixel, xmaxPixel, ymaxPixel = int(xminPixel), int(yminPixel), int(xmaxPixel), int(ymaxPixel)
        cluster_nightlights.append(tif_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean())

    df['nightlights'] = cluster_nightlights

def generate_download_locations(df, ipc=50):
    '''
    Takes a dataframe with columns cluster_lat, cluster_lon
    Generates a 10km x 10km bounding box around the cluster and samples
    ipc images per cluster. First samples in a grid fashion, then any
    remaining points are randomly (uniformly) chosen
    '''
    np.random.seed(RANDOM_SEED) # for reproducability
    df_download = {'image_name': [], 'image_lat': [], 'image_lon': [], 'cluster_lat': [],
                   'cluster_lon': [], 'cons_pc': [], 'nightlights': [] }

    # side length of square for uniform distribution
    edge_num = math.floor(math.sqrt(ipc))
    for _, r in df.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)
        lats = np.linspace(min_lat, max_lat, edge_num).tolist()
        lons = np.linspace(min_lon, max_lon, edge_num).tolist()

        # performs cartesian product
        uniform_points = np.transpose([np.tile(lats, len(lons)), np.repeat(lons, len(lats))])

        lats = uniform_points[:,0].tolist()
        lons = uniform_points[:,1].tolist()

        # fills the remainder with random points
        for _ in range(ipc - edge_num * edge_num):
            lat = random.uniform(min_lat, max_lat)
            lon = random.uniform(min_lon, max_lon)
            lats.append(lat)
            lons.append(lon)

        # add to dict
        for lat, lon in zip(lats, lons):
            # image name is going to be image_lat_image_lon_cluster_lat_cluster_lon.png
            image_name = str(lat) + '_' + str(lon) + '_' + str(r.cluster_lat) + '_' + str(r.cluster_lon) + '.png'
            df_download['image_name'].append(image_name)
            df_download['image_lat'].append(lat)
            df_download['image_lon'].append(lon)
            df_download['cluster_lat'].append(r.cluster_lat)
            df_download['cluster_lon'].append(r.cluster_lon)
            df_download['cons_pc'].append(r.cons_pc)
            df_download['nightlights'].append(r.nightlights)

    return pd.DataFrame.from_dict(df_download)


def drop_0s(df, fr=0.1):
    """
        Solves for d:
            (c_z - d)/(n - d) = fr
        Where d = rows to drop, c_z = num rows with zero nightlights, n = num rows, fr = frac remaining

        Yields:
        d = (c_z - n*fr) / (1 - fr)
    """
    np.random.seed(RANDOM_SEED)
    c_z = (df['nightlights']==0).sum()
    n = len(df)
    assert c_z / n > fr, print(f'Dataframe already has under {fr} zeros')

    d = (c_z - n * fr) / (1 - fr)
    d = int(d)
    print(f'dropping: {d}')

    zero_df = df[df['nightlights']==0]
    zero_clusters = zero_df.groupby(['cluster_lat', 'cluster_lon'])
    per_cluster_drop = int(d / len(zero_clusters))
    print(f'Need to drop {per_cluster_drop} per cluster with 0 nightlights')

    drop_inds = []
    for (cluster_lat, cluster_lon), group in zero_clusters:
        z_inds = group.index
        clust_drop = np.random.choice(z_inds, per_cluster_drop, replace=False)
        assert len(group) - len(clust_drop) >= 10, print(f'dropping too many in {cluster_lat}, {cluster_lon}')
        drop_inds += clust_drop.tolist()

    # this is how you do it purely randomly but some clusters might get wiped out
    # z_inds = np.argwhere(df['nightlights'].values == 0).reshape(-1)
    # drop_inds = np.random.choice(z_inds, d, replace=False)
    return df.drop(drop_inds).reset_index(drop=True)


def drop_in_range(df, lower=0, upper=2, fr=0.25):
    """
        Very similar to drop_0s calculation, but more generalized. Lower and upper are inclusive.
    """
    np.random.seed(RANDOM_SEED)
    boolean_idx = ((lower <= df['nightlights']) & (df['nightlights'] <= upper))
    c_under = boolean_idx.sum()
    n = len(df)
    assert c_under / n > fr, print(f'Dataframe already has under {fr} rows in the given range')

    d = (c_under - n * fr) / (1 - fr)
    d = int(d)
    print(f'dropping: {d}')

    select_df = df[boolean_idx]
    select_clusters = select_df.groupby(['cluster_lat', 'cluster_lon'])
    per_cluster_drop = int(d / len(select_clusters))
    print(f'Need to drop {per_cluster_drop} per cluster in the given range')

    drop_inds = []
    for (cluster_lat, cluster_lon), group in select_clusters:
        z_inds = group.index
        clust_drop = np.random.choice(z_inds, per_cluster_drop, replace=False)
        assert len(group) - len(clust_drop) >= 10, print(f'dropping too many in {cluster_lat}, {cluster_lon}')
        drop_inds += clust_drop.tolist()

    return df.drop(drop_inds).reset_index(drop=True)


main()
