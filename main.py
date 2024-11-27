from typing import Dict, Tuple
import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import mapping, shape
from rasterio.transform import from_bounds
import json
import traceback
import firebase_admin
from firebase_admin import credentials, db
import os
from google.cloud import secretmanager

def get_firebase_credentials():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

def initialize_firebase():
    try:
        # Get the credentials JSON from Google Secret Manager
        credentials_json = get_firebase_credentials()
        # cred_path = r'C:\Users\nyderl\Documents\Github\GeospatialApi/serviceAccountKey.json'
        # cred = credentials.Certificate(cred_path)

        # Check if the Firebase app is already initialized
        if not firebase_admin._apps:
            # Initialize with the JSON string
            cred = credentials.Certificate(json.loads(credentials_json))
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://nydalprojects-default-rtdb.europe-west1.firebasedatabase.app/'
            })
    except Exception as e:
        raise ValueError(f"Failed to initialize Firebase: {str(e)}")


def ensure_properties_in_geojson(geo_dict: Dict) -> Dict:
    """
    Ensures every feature in the GeoJSON contains a 'properties' field.
    If the 'properties' field is missing, an empty dictionary is added.
    """
    try:
        # Ensure 'properties' exists for each feature in building_limits
        if 'features' in geo_dict['building_limits']:
            for feature in geo_dict['building_limits']['features']:
                if 'properties' not in feature:
                    feature['properties'] = {}

        # Ensure 'properties' exists for each feature in height_plateaus
        if 'features' in geo_dict['height_plateaus']:
            for feature in geo_dict['height_plateaus']['features']:
                if 'properties' not in feature:
                    feature['properties'] = {}

    except KeyError as e:
        raise ValueError(f"Error in GeoJSON structure: {str(e)}")
    
    return geo_dict



def read_geojson_from_firebase() -> Dict:
    try:
        # Reference to the root of the database
        ref = db.reference('/')

        # Fetch the data
        data = ref.get()

        # Ensure data contains 'building_limits' and 'height_plateaus'
        if 'building_limits' not in data or 'height_plateaus' not in data:
            raise ValueError("Database must contain 'building_limits' and 'height_plateaus' keys.")


        data = ensure_properties_in_geojson(data)

        geo_dict = {
            'building_limits': data['building_limits'],
            'height_plateaus': data['height_plateaus']
        }

        return geo_dict
    except Exception as e:
        raise ValueError(f"Failed to read data from Firebase: {str(e)}")

def write_polygons_to_firebase(
    building_limits_gdf: gpd.GeoDataFrame,
    height_plateaus_gdf: gpd.GeoDataFrame
):
    try:
        # Convert GeoDataFrames to GeoJSON-like dictionaries
        building_limits_features = json.loads(building_limits_gdf.to_json())['features']
        height_plateaus_features = json.loads(height_plateaus_gdf.to_json())['features']

        # Reference to the root of the database
        ref = db.reference('/')

        # Prepare the data to be written
        data = {
            'building_limits': building_limits_features,
            'height_plateaus': height_plateaus_features
        }

        # Write data to Firebase
        ref.set(data)
    except Exception as e:
        raise ValueError(f"Failed to write polygons to Firebase: {str(e)}")

def read_geojson_create_geodataframe(geo_dict: Dict) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    try:

        print(geo_dict)
        
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
) -> Tuple[xr.DataArray, Tuple[float, float, float, float], gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
        resolution = 0.000005  # Increased precision (approx ~0.5 meters at the equator)
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        # Warn if the raster size is very large
        total_cells = width * height
        if total_cells > 1e8:
            print(f"Warning: The raster size is very large ({total_cells} cells).")
            print("This may consume a lot of memory and take a long time to process.")

        # Define transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Rasterize building limits
        building_limits_raster = rasterio.features.rasterize(
            ((geom, 1) for geom in building_limits_gdf.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='int8',
            all_touched=True
        )

        # Initialize height_plateaus_raster with zeros
        height_plateaus_raster = np.zeros((height, width), dtype='float32')

        # Rasterize each height plateau and use maximum value in case of overlaps
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
            height_plateaus_raster = np.maximum(height_plateaus_raster, raster)

        # Create xarray DataArray
        x_coords = np.linspace(minx, maxx, width)
        y_coords = np.linspace(maxy, miny, height)  # Note: maxy to miny to reverse y-axis

        height_da = xr.DataArray(
            height_plateaus_raster,
            coords=[y_coords, x_coords],
            dims=["y", "x"]
        )

        # Mask areas outside building limits
        building_mask = building_limits_raster == 1

        # Set areas outside building limits to NaN
        height_da = height_da.where(building_mask, other=np.nan)

        # Set areas inside building limits but not covered by any height plateau to 0
        inside_building_no_plateau = (building_mask) & (np.isnan(height_da))
        height_da = height_da.where(~inside_building_no_plateau, other=0)

        # Reproject GeoDataFrames to match the raster CRS if needed
        raster_crs = rasterio.crs.CRS.from_epsg(4326)  # Assuming EPSG:4326
        if building_limits_gdf.crs != raster_crs:
            building_limits_gdf = building_limits_gdf.to_crs(raster_crs)
        if height_plateaus_gdf.crs != raster_crs:
            height_plateaus_gdf = height_plateaus_gdf.to_crs(raster_crs)

        return height_da, (minx, miny, maxx, maxy), building_limits_gdf, height_plateaus_gdf
    except Exception as e:
        raise ValueError(f"Failed to rasterize GeoDataFrames: {str(e)}")


def add_height_plateau_to_firebase(geometry: Dict, height: float):
    """
    Adds a new height plateau to the Firebase database.
    
    Parameters:
    - geometry: The GeoJSON geometry of the height plateau.
    - height: The height value for the plateau.
    """
    try:
        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Add the new plateau
        new_plateau = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {"height": height}
        }
        geo_dict["height_plateaus"]["features"].append(new_plateau)

        # Write back to Firebase
        write_polygons_to_firebase(
            building_limits_gdf=gpd.GeoDataFrame.from_features(
                geo_dict["building_limits"]["features"], crs="epsg:4326"
            ),
            height_plateaus_gdf=gpd.GeoDataFrame.from_features(
                geo_dict["height_plateaus"]["features"], crs="epsg:4326"
            )
        )

        print("Height plateau added successfully!")
    except Exception as e:
        print(f"Failed to add height plateau: {str(e)}")


def delete_height_plateau_from_firebase(target_height: float):
    """
    Deletes a specific height plateau with a given height value from the Firebase database.
    
    Parameters:
    - target_height: The height value of the plateau to delete.
    """
    try:
        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Filter out the plateau with the target height
        filtered_features = [
            feature for feature in geo_dict["height_plateaus"]["features"]
            if feature["properties"].get("height") != target_height
        ]

        # Update the height plateaus
        geo_dict["height_plateaus"]["features"] = filtered_features

        # Write back to Firebase
        write_polygons_to_firebase(
            building_limits_gdf=gpd.GeoDataFrame.from_features(
                geo_dict["building_limits"]["features"], crs="epsg:4326"
            ),
            height_plateaus_gdf=gpd.GeoDataFrame.from_features(
                geo_dict["height_plateaus"]["features"], crs="epsg:4326"
            )
        )

        print(f"Height plateau with height {target_height} deleted successfully!")
    except Exception as e:
        print(f"Failed to delete height plateau: {str(e)}")


def modify_building_limits_in_firebase(new_geometry: Dict):
    """
    Modifies the building limits in the Firebase database.

    Parameters:
    - new_geometry: The new GeoJSON geometry for the building limits.
    """
    try:
        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Update the building limits
        geo_dict["building_limits"]["features"] = [
            {
                "type": "Feature",
                "geometry": new_geometry,
                "properties": {}
            }
        ]

        # Write back to Firebase
        write_polygons_to_firebase(
            building_limits_gdf=gpd.GeoDataFrame.from_features(
                geo_dict["building_limits"]["features"], crs="epsg:4326"
            ),
            height_plateaus_gdf=gpd.GeoDataFrame.from_features(
                geo_dict["height_plateaus"]["features"], crs="epsg:4326"
            )
        )

        print("Building limits updated successfully!")
    except Exception as e:
        print(f"Failed to modify building limits: {str(e)}")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import geopandas as gpd

# Firebase Initialization
initialize_firebase()

# FastAPI Application
app = FastAPI()

# Pydantic models for request validation
class GeoJSON(BaseModel):
    type: str
    geometry: Dict
    properties: Optional[Dict] = {}

class ModifyRequest(BaseModel):
    new_geometry: Dict

class AddHeightPlateauRequest(BaseModel):
    geometry: Dict
    height: float

class DeleteHeightPlateauRequest(BaseModel):
    height: float


@app.get("/")
def root():
    return {"message": "Geospatial API is running."}


@app.get("/read")
def read_data():
    """Read data from Firebase."""
    try:
        geo_dict = read_geojson_from_firebase()
        return geo_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data: {str(e)}")


@app.post("/add-height-plateau")
def add_height_plateau(request: AddHeightPlateauRequest):
    """Add a new height plateau."""
    try:
        add_height_plateau_to_firebase(request.geometry, request.height)
        return {"message": "Height plateau added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add height plateau: {str(e)}")


@app.delete("/delete-height-plateau")
def delete_height_plateau(request: DeleteHeightPlateauRequest):
    """Delete a height plateau with a specific height."""
    try:
        delete_height_plateau_from_firebase(request.height)
        return {"message": f"Height plateau with height {request.height} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete height plateau: {str(e)}")


@app.put("/modify-building-limits")
def modify_building_limits(request: ModifyRequest):
    """Modify the building limits."""
    try:
        modify_building_limits_in_firebase(request.new_geometry)
        return {"message": "Building limits updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify building limits: {str(e)}")


@app.put("/rasterize")
def rasterize():
    """Rasterize GeoDataFrames and return visualization data."""
    try:
        geo_dict = read_geojson_from_firebase()
        building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)
        height_da, bounds, _, _ = rasterize_geodataframes(building_limits_gdf, height_plateaus_gdf)
        return {
            "height_da": height_da.values.tolist(),
            "bounds": bounds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rasterize: {str(e)}")


@app.get("/visualize")
def visualize():
    """Visualize the building limits and height plateaus."""
    try:
        geo_dict = read_geojson_from_firebase()
        building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)
        height_da, bounds, _, _ = rasterize_geodataframes(building_limits_gdf, height_plateaus_gdf)
        visualize_height_da(height_da, building_limits_gdf, height_plateaus_gdf, bounds)
        return {"message": "Visualization completed. Check your plot window."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to visualize: {str(e)}")




def main():
    try:
        # Initialize Firebase
        initialize_firebase()
        # Read the GeoJSON data from Firebase
        geo_dict = read_geojson_from_firebase()

        # Read GeoDataFrames
        building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)

        # Ensure 'elevation' column exists in height_plateaus_gdf
        if 'elevation' not in height_plateaus_gdf.columns:
            raise ValueError("Height plateaus GeoDataFrame must have an 'elevation' column.")

        # Optionally, update or process the GeoDataFrames here...
        # For example, apply some transformations or filters

        # Rasterize GeoDataFrames
        height_da, bounds, building_limits_gdf, height_plateaus_gdf = rasterize_geodataframes(
            building_limits_gdf, height_plateaus_gdf
        )

        # Write updated polygons back to Firebase
        # write_polygons_to_firebase(building_limits_gdf, height_plateaus_gdf)

        # Visualize the DataArray with building limits and height plateaus boundaries
        visualize_height_da(height_da, building_limits_gdf, height_plateaus_gdf, bounds)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred: {str(e)}\n\n{tb}")

if __name__ == "__main__":
    main()
