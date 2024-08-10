import os, geopandas as gpd, shapely
import pandas as pd
import boto3

import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.io import MemoryFile
import urllib.parse as urlparse

import numpy as np
from .utils import reads3csv_with_credential
from scipy.sparse.csgraph import connected_components


def get_grid_from_centroid(centroid, width=0.0025, height=0.0025, crs_old=4326, crs_new=4326):
    gf = gpd.GeoDataFrame({
        'lat': centroid[0],
        'lon': centroid[1],
        'width': width,
        'height': height
    }, index=[0])

    gf.crs = {'init': 'epsg:{}'.format(crs_old)}

    gf['center'] = gf.apply(lambda x: shapely.geometry.Point(x['lat'], x['lon']), axis=1)

    gf = gf.set_geometry('center')
    gf = gf.to_crs(epsg=crs_new)

    # create polygon using width and height
    gf['center'] = gf['center'].buffer(1)
    gf['polygon'] = gf.apply(lambda x: shapely.affinity.scale(x['center'], height, width), axis=1)
    gf = gf.set_geometry('polygon')

    # get bounding box of created polygon
    gf['geometry'] = gf['polygon'].envelope
    gf = gf \
        .set_geometry('geometry') \
        .filter(items=['lat', 'lon', 'geometry'])
    return gf


# three class
def write_threeclass_by_grid(grid_df, col_shp, resolution, diam, crs, buf_dist, dir_out, s3_client):
    # rasterize and write
    centroid = (grid_df['x'], grid_df['y'])

    grid = get_grid_from_centroid(centroid, diam, diam, crs, crs)

    # set metadata
    minx, miny, maxx, maxy = grid.total_bounds
    shape = (int(round((maxx - minx) / resolution)), int(round((maxy - miny) / resolution)))

    transform = (resolution, 0.0, minx, 0.0, -resolution, maxy)
    meta = ({
        'driver': 'GTiff',
        'dtype': 'int16',
        'nodata': None,
        'width': shape[0],
        'height': shape[1],
        'count': 1,
        'crs': CRS.from_epsg(crs),
        'transform': transform

    })

    # get shape
    shp = gpd.read_file(grid_df[col_shp])
    shp['category'] = 1
    shp['buffer_in'] = shp.geometry.buffer(buf_dist)
    shp['buffer_out'] = shp.geometry.buffer(-buf_dist)
    shp = gpd.overlay(grid, shp, how='intersection')

    out_fn = "{}.tif".format(grid_df['name_col_row'])
    # print(shape)
    out_arr = np.zeros(shape).astype('int16')
    if len(shp) > 0:
        print(out_fn)

        shapes = ((geom, value) for geom, value in zip(shp['geometry'], shp['category']))
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr.copy(), transform=meta['transform'])

        try:
            shapes_shrink = ((geom, value) for geom, value in zip(shp['buffer_in'], shp['category']))
            shrinked = features.rasterize(shapes=shapes_shrink, fill=0, out=out_arr.copy(), transform=meta['transform'])
            shapes_explode = ((geom, value) for geom, value in zip(shp['buffer_out'], shp['category']))
            exploded = features.rasterize(shapes=shapes_explode, fill=0, out=out_arr.copy(),
                                          transform=meta['transform'])
        except:
            shp['buffer'] = shp.geometry.buffer(-1 * resolution)
            shapes_shrink = ((geom, value) for geom, value in zip(shp['buffer'], shp['category']))
            shrinked = features.rasterize(shapes=shapes_shrink, fill=0, out=out_arr.copy(), transform=meta['transform'])

        out = burned * 2 - shrinked + np.where((exploded * 2 - burned) == 1, 0, exploded * 2 - burned).astype(
            np.int16)
    else:
        out = out_arr

    if dir_out.startswith("s3"):

        dir_out_parsed = urlparse.urlparse(dir_out)
        bucket = dir_out_parsed.netloc
        prefix = dir_out_parsed.path
        with MemoryFile() as memfile:
            with memfile.open(**meta) as src:
                src.write(out, 1)
            s3_client.upload_fileobj(Fileobj=memfile,
                                     Bucket=bucket,
                                     Key=os.path.join(prefix + out_fn))
    else:
        with rasterio.open(os.path.join(dir_out, out_fn), "w+", **meta) as dst:
            dst.write_band(1, out)


# Binary
def write_binary_by_grid(grid_df, col_shp, resolution, diam, crs, dir_out, s3_client):
    # rasterize and write
    centroid = (grid_df['x'], grid_df['y'])
    grid = get_grid_from_centroid(centroid, diam, diam, crs, crs)

    # set metadata
    minx, miny, maxx, maxy = grid.total_bounds
    shape = (int(round((maxx - minx) / resolution)), int(round((maxy - miny) / resolution)))

    transform = (resolution, 0.0, minx, 0.0, -resolution, maxy)
    meta = ({
        'driver': 'GTiff',
        'dtype': 'int16',
        'nodata': None,
        'width': shape[0],
        'height': shape[1],
        'count': 1,
        'crs': CRS.from_epsg(crs),
        'transform': transform

    })

    shp = gpd.read_file(grid_df[col_shp])
    shp['category'] = 1
    shp = gpd.overlay(grid, shp, how='intersection')
    out_fn = "{}.tif".format(grid_df['name_col_row'])
    out_arr = np.zeros(shape).astype('int16')
    if len(shp) > 0:
        print(out_fn)
        shapes = ((geom, value) for geom, value in zip(shp['geometry'], shp['category']))
        out = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=meta['transform'])
    else:
        out = out_arr

    # write
    if dir_out.startswith("s3"):

        dir_out_parsed = urlparse.urlparse(dir_out)
        bucket = dir_out_parsed.netloc
        prefix = dir_out_parsed.path
        with MemoryFile() as memfile:
            with memfile.open(**meta) as src:
                src.write(out, 1)
            s3_client.upload_fileobj(Fileobj=memfile,
                                     Bucket=bucket,
                                     Key=os.path.join(prefix + out_fn))
    else:
        with rasterio.open(os.path.join(dir_out, out_fn), "w+", **meta) as dst:
            dst.write_band(1, out)


def get_rasterization(params, run_local):

    mode = params['raster_mode']
    assert mode in ['three_class', 'binary']
    dir_grids = params['dir_grids']
    dir_catalog = params['dir_catalog']
    col_shp = params['col_shapefile']
    dir_out = params['dir_out']
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    # params concerning raster
    rst_res = params['resolution']
    diam = params['diam']
    crs_epsg = params['crs_epsg']
    if run_local:
        ACCESS_KEY_ID = params['aws_access']
        SECRET_ACCESS_KEY = params['aws_secret']
        REGION = params['aws_region']

        catalog = reads3csv_with_credential(dir_catalog, ACCESS_KEY_ID, SECRET_ACCESS_KEY) \
            if dir_catalog.startswith("s3") else pd.read_csv(dir_catalog)
        grids = reads3csv_with_credential(dir_grids, ACCESS_KEY_ID, SECRET_ACCESS_KEY) \
            if dir_grids.startswith("s3") else pd.read_csv(dir_grids) \
            .merge(catalog, how='inner', on=['name'])
        # in case the i
        s3_client = boto3.client("s3",
                                 aws_access_key_id=ACCESS_KEY_ID,
                                 aws_secret_access_key=SECRET_ACCESS_KEY,
                                 region_name=REGION
                                 )
        if mode == 'three_class':
            grids.apply(lambda x: write_threeclass_by_grid(x, col_shp, rst_res, diam, crs_epsg, -1 * rst_res,
                                                               dir_out, s3_client),
                        axis=1)
        else:
            # print(grid_df, col_shp, resolution, diam, crs, dir_out, s3_client)
            grids.apply(
                lambda x: write_binary_by_grid(x, col_shp, rst_res, diam, crs_epsg, dir_out, s3_client),
                axis=1)
    else:
        catalog = pd.read_csv(dir_catalog)
        grids = pd.read_csv(dir_grids)\
            .merge(catalog, how='inner', on=['name'])
        if mode == 'three_class':
            grids.apply(
                lambda x: write_threeclass_by_grid(x, col_shp, rst_res, diam, crs_epsg, -1 * rst_res, dir_out,
                                                       None),
                axis=1)
        else:
            grids.apply(
                lambda x: write_binary_by_grid(x, col_shp, rst_res, diam, crs_epsg, dir_out,
                                                   None),
                axis=1)




