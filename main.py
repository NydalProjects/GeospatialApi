from typing import Dict, Tuple, Union
import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from rasterio.transform import from_bounds
import json
import traceback

def read_geojson_create_geodataframe(geo_dict: Dict) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    try:
        # Create GeoDataFrame from building_limits feature
        building_limits_gdf = gpd.GeoDataFrame.from_features(
            geo_dict['building_limits'],
            crs='epsg:4326'
        )

        # Create GeoDataFrame from height_plateaus feature
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(
            geo_dict['height_plateaus'],
            crs='epsg:4326'
        )
    except KeyError as e:
        raise ValueError("Failed to create GeoDataFrame: {}".format(str(e)))

    return building_limits_gdf, height_plateaus_gdf

def rasterize_geodataframes(
    building_limits_gdf: gpd.GeoDataFrame,
    height_plateaus_gdf: gpd.GeoDataFrame
) -> Tuple[xr.DataArray, Tuple[float, float, float, float], gpd.GeoDataFrame]:
    try:
        # Define the output raster parameters
        # Get combined bounds
        minx, miny, maxx, maxy = building_limits_gdf.total_bounds
        h_minx, h_miny, h_maxx, h_maxy = height_plateaus_gdf.total_bounds
        minx = min(minx, h_minx)
        miny = min(miny, h_miny)
        maxx = max(maxx, h_maxx)
        maxy = max(maxy, h_maxy)

        # Define resolution (adjust as needed)
        resolution = 0.0001  # Approx ~11 meters at the equator
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        # Define transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Rasterize building limits
        building_limits_raster = rasterio.features.rasterize(
            ((geom, 1) for geom in building_limits_gdf.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='int16',
            all_touched=True
        )

        # Initialize height_plateaus_raster with zeros
        height_plateaus_raster = np.zeros((height, width), dtype='float32')

        # Rasterize each height plateau and sum their values
        for idx, row in height_plateaus_gdf.iterrows():
            elevation = row.get('elevation', 0)
            geom = row.geometry
            if geom.is_empty:
                continue
            raster = rasterio.features.rasterize(
                [(geom, elevation)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='float32',
                all_touched=True
            )
            height_plateaus_raster += raster

        # Create xarray DataArray
        x_coords = np.linspace(minx, maxx, width)
        y_coords = np.linspace(miny, maxy, height)

        height_da = xr.DataArray(
            height_plateaus_raster,
            coords=[y_coords, x_coords],
            dims=["y", "x"]
        )

        # Mask areas outside building limits
        building_mask = building_limits_raster == 1

        # Set areas outside building limits to -1
        height_da = height_da.where(building_mask, other=-1)

        # Set areas inside building limits but not covered by any height plateau to 0
        inside_building_no_plateau = (building_mask) & (height_da == 0)
        height_da = height_da.where(~inside_building_no_plateau, other=0)

        # Reproject building_limits_gdf to match the raster CRS if needed
        raster_crs = rasterio.crs.CRS.from_epsg(4326)  # Assuming EPSG:4326
        if building_limits_gdf.crs != raster_crs:
            building_limits_gdf = building_limits_gdf.to_crs(raster_crs)

        return height_da, (minx, miny, maxx, maxy), building_limits_gdf
    except Exception as e:
        raise ValueError(f"Failed to rasterize GeoDataFrames: {str(e)}")

def visualize_height_da(height_da: xr.DataArray, building_limits_gdf: gpd.GeoDataFrame, bounds: Tuple[float, float, float, float]):
    try:
        # Plot the DataArray
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_under(color='white')  # For values less than vmin

        # Plot settings
        im = height_da.plot.imshow(
            ax=ax,
            cmap=cmap,
            vmin=0,  # Values less than 0 will use 'under' color
            add_colorbar=True,
            cbar_kwargs={'label': 'Height (units)'}
        )

        # Overlay the building limits boundary
        building_limits_gdf.boundary.plot(
            ax=ax,
            edgecolor='black',  # Border line color
            linewidth=1.5,       # Border line width
            linestyle='--',      # Border line style (dashed)
            label='Building Limits'
        )

        # Set plot limits to match the raster bounds
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Customize the plot
        ax.set_title('Combined Height Plateaus and Building Limits')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Add legend
        ax.legend()

        plt.show()
    except Exception as e:
        raise ValueError(f"Failed to visualize height DataArray: {str(e)}")

def main():
    try:
        # Read the JSON data from file
        with open('shapes.json', 'r') as f:
            geo_dict = json.load(f)

        # Read GeoDataFrames
        building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)

        # Ensure 'elevation' column exists in height_plateaus_gdf
        if 'elevation' not in height_plateaus_gdf.columns:
            raise ValueError("Height plateaus GeoDataFrame must have an 'elevation' column.")

        # Rasterize GeoDataFrames
        height_da, bounds, building_limits_gdf = rasterize_geodataframes(building_limits_gdf, height_plateaus_gdf)

        # Visualize the DataArray with building limits boundary
        visualize_height_da(height_da, building_limits_gdf, bounds)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred: {str(e)}\n\n{tb}")

if __name__ == "__main__":
    main()
