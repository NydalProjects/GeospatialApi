import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Dict, Tuple
import requests

BASE_URL = "https://my-cloud-run-service-861222091615.us-central1.run.app"

def rasterize():

    """Fetch rasterized data from the API."""

    response = requests.put(f"{BASE_URL}/rasterize")
    print(response)
    if response.status_code == 200:
        print("Rasterized data retrieved successfully.")
        rasterized_data = response.json()

        # Deserialize the rasterized data into an xarray DataArray
        height_da = xr.DataArray(
            np.array(rasterized_data["height_da"]),  # Convert list back to NumPy array
            coords=[
                rasterized_data["y_coords"],  # y-coordinates
                rasterized_data["x_coords"],  # x-coordinates
            ],
            dims=["y", "x"],  # Dimension names
        )

        bounds = rasterized_data["bounds"]  # Bounds for further use
        return height_da, bounds
    else:
        print(f"Failed to rasterize: {response.text}")
        return None


def visualize_rasterized_data(height_da: Dict, bounds):
    """Visualize the rasterized data."""
    try:
        # Extract data

        # Fetch GeoJSON data for overlay
        geo_response = requests.get(f"{BASE_URL}/read")
        if geo_response.status_code != 200:
            print(f"Failed to fetch GeoJSON data: {geo_response.text}")
            return
        geo_data = geo_response.json()

        # Convert GeoJSON to GeoDataFrames
        building_limits_gdf = gpd.GeoDataFrame.from_features(geo_data["building_limits"]["features"], crs="epsg:4326")
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(geo_data["height_plateaus"]["features"], crs="epsg:4326")
        print(building_limits_gdf)
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


def update_firebase():
    """Update Firebase with modified building limits and height plateaus."""
    # Correctly structured building limits as a FeatureCollection
    new_building_limits = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [10.7579, 59.9134],
                            [10.7566, 59.9137],
                            [10.7564, 59.9135],
                            [10.7563, 59.9133],
                            [10.7561, 59.9129],
                            [10.7563, 59.9129],
                            [10.7575, 59.9128],
                            [10.7579, 59.9134]
                        ]
                    ]
                },
                "properties": {}
            }
        ]
    }

    # Correctly structured height plateaus as a FeatureCollection
    new_height_plateaus = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [10.7568, 59.9129],
                            [10.7575, 59.9128],
                            [10.7579, 59.9134],
                            [10.7573, 59.9135],
                            [10.7568, 59.9129]
                        ]
                    ]
                },
                "properties": {"elevation": 3.75}
            },
            # Add additional features as required
        ]
    }

    # Update building limits
    response = requests.put(
        f"{BASE_URL}/modify-building-limits",
        json={"new_geometry": new_building_limits["features"][0]["geometry"]}
    )

    if response.status_code == 200:
        print("Building limits updated successfully.")
    else:
        print(f"Failed to update building limits: {response.text}")

    # Update height plateaus
    for plateau in new_height_plateaus["features"]:
        response = requests.post(
            f"{BASE_URL}/add-height-plateau",
            json={"geometry": plateau["geometry"], "height": plateau["properties"]["elevation"]}
        )
        if response.status_code == 200:
            print("Height plateau added successfully.")
        else:
            print(f"Failed to add height plateau: {response.text}")



# if __name__ == "__main__":
#     update_firebase()

if __name__ == "__main__":
    # Fetch and visualize rasterized data
    rasterized_data, bounds = rasterize()
    print(rasterized_data)
    print(bounds)
    if rasterized_data is not None and rasterized_data.size > 0:
        visualize_rasterized_data(rasterized_data, bounds)