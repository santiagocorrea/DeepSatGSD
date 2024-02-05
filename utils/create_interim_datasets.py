#!/usr/bin/env python3
"""
Module to create the interim datasets which crops the mosaics based on sensor
"""

__author__ = "Santiago Correa"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import pdb
import pandas as pd
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import os
from tqdm import tqdm  # Import tqdm for the progress bar

def main(args):
    """ Main entry point of the app """
    print(args)
    meta_df = gpd.read_file(f"/work/scorreacardo_umass_edu/DeepSatGSD/data/interim/dg_metadata_sensorcount{args.number}.gpkg")
    output_dir = "/work/scorreacardo_umass_edu/DeepSatGSD/data/interim"
    input_dir = "/gypsum/scratch1/jtaneja/DG/DG_new"
    #pdb.set_trace()
    # Define the total number of iterations
    total_iterations = len(meta_df)
    # Loop through the DataFrame rows
    for index, row in tqdm(meta_df.iterrows(), total=total_iterations, desc='Processing'):
        # Extract the file name, geometry, and sensor
        filename = row['FILENAME']
        geometry = row['geometry']
        sensor = row['SENSOR']
        date = row['ACQ_DATE']
        #tqdm.set_description(f'Processing: {filename}') 

        # Load the TIFF file
        with rasterio.open(os.path.join(input_dir, filename)) as src:
            # Mask the TIFF file based on the geometry
            masked_data, masked_transform = mask(src, [geometry], crop=True)
            masked_meta = src.meta

        # Set the output file path
        output_subdir = os.path.join(output_dir, sensor)
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = os.path.join(output_subdir, os.path.basename(filename)[:-4] + f"_{str(date)}" + f"_{sensor}" +".tif")

        # Update the metadata for the masked TIFF
        masked_meta.update({
            'transform': masked_transform,
            'height': masked_data.shape[1],
            'width': masked_data.shape[2]
        })

        # Save the masked TIFF file
        with rasterio.open(output_filename, 'w', **masked_meta) as dst:
            dst.write(masked_data)



if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Optional argument flag which defaults to False
    parser.add_argument("-f", "--flag", action="store_true", default=False)

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--number", action="store", dest="number")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
