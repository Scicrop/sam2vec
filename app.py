import argparse
import glob
import os
import leafmap
import requests
import rasterio
import numpy as np
from rasterio.enums import Resampling
from samgeo import SamGeo, tms_to_geotiff, get_basemaps
from decouple import config
import geopandas as gpd
import matplotlib.pyplot as plt

api_root = 'https://app.agroapi.com.br'
api_token = config('TOKEN')


def get_sentinel2_images_dates(geometry, cloud_percentage):
    print('Getting Sentinel 2 image from geometry')
    url = f'{api_root}/api/data/image/satellite/aoi/dates'
    data = {
        "aoi": {"type": "Feature", "geometry": geometry},
        "date": "9999-01-01",
        "source": "SENTINEL-2",
        "offset": 0,
        "limit": 24
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token ' + api_token
    }
    response = requests.post(url, json=data, headers=headers)
    results = response.json()["data"]["results"]
    dates = []
    for result in results:
        if result["cloud"] == cloud_percentage:
            dates.append(result["date"])
    return dates


def car_to_geojson(car):
    print('Getting geometry from CAR')
    url = f'{api_root}/api/data/environmental/car/imovel/codigo-imovel/{car}'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token ' + api_token
    }
    response = requests.get(url, headers=headers)
    return response.json()['data']['geometry']


def sum_ndvi_images(tiff_paths, output_path):
    sum_array = None

    # Loop through the list of TIFF file paths
    for path in tiff_paths:
        print(path)
        with rasterio.open(path) as src:
            image_array = src.read(1)

            if sum_array is None:
                sum_array = np.zeros_like(image_array, dtype=np.float64)

            sum_array += image_array

    with rasterio.open(tiff_paths[0]) as src:
        meta = src.meta.copy()

    meta.update(dtype=rasterio.float64)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(sum_array, 1)

    print(f"Summed NDVI image saved to {output_path}")


def sentinel2_geo_tiff_from_geojson(geometry, band_type, destination_path, date):
    url = f'{api_root}/api/data/image/satellite/aoi'
    data = {
        "aoi":
            {
                "type": "Feature", "geometry": geometry},
        "date": date,
        "source": "SENTINEL-2",
        "mode": band_type,
        "crop": True
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'image/tiff',
        'Authorization': f'Token ' + api_token
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(destination_path, 'wb') as f:
            f.write(response.content)
        print(f'TIFF image saved to {destination_path}')
    else:
        print(f'Failed to retrieve image. Status code: {response.status_code}, Response: {response.text}')


def convert_band_to_uint8(src_path, dest_path):
    with rasterio.open(src_path) as src:
        profile = src.profile
        band = src.read(1)
        nodata_val = src.nodata
        band[np.isnan(band)] = 0
        #band[band < 0] = 0
        band[band > 1] = 255

        band_scaled = (band * 255).astype(np.uint8)

        profile.update(dtype=rasterio.uint8, nodata=None)

        with rasterio.open(dest_path, 'w', **profile) as dst:
            dst.write(band_scaled, 1)

    print(f'Converted image saved to {dest_path}')


def init_sam():
    print('Initializing SAM')
    sam = SamGeo(
        model_type="vit_h",
        checkpoint="sam_vit_h_4b8939.pth",
        sam_kwargs=None,
    )
    return sam


def plot_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    gdf.plot()
    plt.title("Shapefile Plot")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


def plot_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        img = src.read()
        if img.shape[0] == 1:
            plt.imshow(img[0], cmap='gray')
            plt.title('Single-band TIFF Image')
        elif img.shape[0] == 3:
            img_rgb = np.dstack((img[0], img[1], img[2]))
            plt.imshow(img_rgb)
            plt.title('RGB TIFF Image')
        else:
            print("Unsupported number of bands:", img.shape[0])
            return

        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.grid(True)
        plt.show()


def process(sam, band_type, image_path):
    print('Generating SAM masks')
    mask = "results/segment.tiff"

    if band_type != 'RGB':
        with rasterio.open(image_path) as src:
            band = src.read(1)
            profile = src.profile
            transform = src.transform
            crs = src.crs

            band_normalized = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
            band_3d = np.stack((band_normalized,) * 3, axis=-1)

        new_image_path = f"results/{band_type}_normalized.tiff"
        new_profile = profile.copy()
        new_profile.update({
            'count': 3,  # 3 bandas
            'dtype': 'uint8'
        })

        with rasterio.open(new_image_path, 'w', **new_profile) as dst:
            for i in range(3):
                dst.write(band_3d[:, :, i], i + 1)

        print(f'Normalized {band_type} image saved to {new_image_path}')
        image_path = new_image_path

    sam.generate(
        image_path, mask, batch=True, foreground=True, erosion_kernel=(3, 3), mask_multiplier=255
    )

    plot_tiff(mask)

    print('Generating GPKG from SAM masks')
    vector = "results/segment.gpkg"
    sam.tiff_to_gpkg(mask, vector, simplify_tolerance=None)
    print('Generating Vector File from SAM mask')
    shapefile = "results/segment.shp"
    sam.tiff_to_vector(mask, shapefile)
    plot_shapefile(shapefile)
    print('Done.')


def rm_unused(directory="results"):
    extensions = ['*.tiff', '*.shp', '*.shx', '*.prj', '*.gpkg', '*.dbf', '*.cpg', '*.tif']
    for ext in extensions:
        files = glob.glob(os.path.join(directory, ext))
        for file in files:
            try:
                os.remove(file)
                print(f'Removed {file}')
            except OSError as e:
                print(f'Error removing {file}: {e.strerror}')


def main(car, date, band):
    rm_unused('results/')
    geometry = car_to_geojson(car)
    destination_path = f"results/{car}"
    final_path = f"results/{car}_final.tiff"
    img_path = f"{destination_path}-{date}-{band}.tiff"
    sentinel2_geo_tiff_from_geojson(geometry, band, img_path, date)
    if band != 'RGB':
        img_rgb_path = f"{destination_path}-{date}-RGB.tiff"
        sentinel2_geo_tiff_from_geojson(geometry, "RGB", img_rgb_path, date)
        plot_tiff(img_rgb_path)
        print(f'Converting {band} image to uint8 with channel dimension')
        convert_band_to_uint8(img_path, img_path)
    plot_tiff(img_path)
    sam = init_sam()
    process(sam, date, img_path)


def list_dates(car):
    geometry = car_to_geojson(car)
    dates = get_sentinel2_images_dates(geometry, 0)
    print(dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Sentinel-2 images and apply SAM.')
    parser.add_argument('--car', required=True, help='CAR code of the property')
    parser.add_argument('--date', help='Date for the satellite image in YYYY-MM-DD format')
    parser.add_argument('--band_type', default='NDVI', choices=['NDVI', 'RGB'], help='Type of image bands to retrieve')
    parser.add_argument('--listdates', action='store_true', help='List available dates for the given CAR')

    args = parser.parse_args()

    if args.listdates:
        if not args.car:
            print('The --car argument is required when using --listdates')
        else:
            list_dates(args.car)
    else:
        if not args.date:
            print('The --date argument is required unless using --listdates')
        else:
            main(args.car, args.date, args.band_type)
