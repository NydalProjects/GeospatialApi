import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Dict, Tuple

BASE_URL = "https://my-cloud-run-service-861222091615.us-central1.run.app"

def rasterize():
    """Fetch rasterized data from the API."""
    response = requests.get(f"{BASE_URL}/")
    print(response)
    exit()

    response = requests.get(f"{BASE_URL}/rasterize")
    print(response)
    if response.status_code == 200:
        print("Rasterized data retrieved successfully.")
        return response.json()
    else:
        print(f"Failed to rasterize: {response.text}")
        return None


def visualize_rasterized_data(rasterized_data: Dict):
    """Visualize the rasterized data."""
    try:
        # Extract data
        height_values = np.array(rasterized_data["height_da"])
        x_coords = rasterized_data["x_coords"]
        y_coords = rasterized_data["y_coords"]
        bounds = rasterized_data["bounds"]

        # Reconstruct the DataArray
        height_da = xr.DataArray(
            height_values,
            coords=[y_coords, x_coords],
            dims=["y", "x"]
        )

        # Fetch GeoJSON data for overlay
        geo_response = requests.get(f"{BASE_URL}/read")
        if geo_response.status_code != 200:
            print(f"Failed to fetch GeoJSON data: {geo_response.text}")
            return
        geo_data = geo_response.json()

        # Convert GeoJSON to GeoDataFrames
        building_limits_gdf = gpd.GeoDataFrame.from_features(geo_data["building_limits"]["features"], crs="epsg:4326")
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(geo_data["height_plateaus"]["features"], crs="epsg:4326")

        # Plot the rasterized data
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='white')  # For NaN values

        # Plot the rasterized height data
        height_da.plot.imshow(
            ax=ax,
            cmap=cmap,
            vmin=np.nanmin(height_da),
            vmax=np.nanmax(height_da),
            add_colorbar=True,
            cbar_kwargs={"label": "Height (units)"},
            origin="lower"
        )

        # Overlay GeoDataFrames
        building_limits_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1, linestyle="--", label="Building Limits")
        height_plateaus_gdf.boundary.plot(ax=ax, edgecolor="blue", linewidth=0.5, alpha=0.7, label="Height Plateaus")

        # Set plot bounds
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Customize the plot
        ax.set_title("Rasterized Height Data")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        plt.show()
    except Exception as e:
        print(f"Failed to visualize rasterized data: {str(e)}")


if __name__ == "__main__":
    # Fetch and visualize rasterized data
    rasterized_data = rasterize()
    if rasterized_data:
        visualize_rasterized_data(rasterized_data)
