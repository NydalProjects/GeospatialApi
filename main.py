from typing import Dict, Tuple
import geopandas as gpd
import rasterio
import rasterio.features
import logging
import numpy as np
import xarray as xr
from shapely.geometry import mapping, shape
from rasterio.transform import from_bounds
import json
import traceback
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import os
from google.cloud import secretmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import geopandas as gpd

# Firebase Initialization

def get_firebase_credentials():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/861222091615/secrets/firebase-secret/versions/latest"
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


# FastAPI Application
app = FastAPI()
initialize_firebase()


def ensure_properties_in_geojson(geo_dict: Dict) -> Dict:
    """
    Ensures every feature in the GeoJSON contains a 'properties' field.
    If the 'properties' field is missing, an empty dictionary is added.
    It also flattens nested 'properties' fields and ensures numeric elevation values.
    """
    try:
        # Process building_limits
        if 'features' in geo_dict.get('building_limits', {}):
            for feature in geo_dict['building_limits']['features']:
                if 'properties' not in feature:
                    feature['properties'] = {}

        # Process height_plateaus
        if 'features' in geo_dict.get('height_plateaus', {}):
            for feature in geo_dict['height_plateaus']['features']:
                if 'properties' not in feature:
                    feature['properties'] = {}

                # Flatten nested properties if present
                if isinstance(feature['properties'], dict) and 'properties' in feature['properties']:
                    feature['properties'].update(feature['properties'].pop('properties'))

                # Ensure elevation is a valid float
                elevation = feature['properties'].get('elevation')
                feature['properties']['elevation'] = float(elevation) if elevation is not None else 0.0


    except KeyError as e:
        raise ValueError(f"Error in GeoJSON structure: {str(e)}")

    return geo_dict


def read_geojson_from_firebase() -> Dict:
    """
    Reads data from Firebase and ensures it conforms to the expected GeoJSON structure.
    """
    try:
        # Reference to the root of the database
        ref = db.reference('/')

        # Fetch the data
        data = ref.get()

        # Validate presence of required keys
        if 'building_limits' not in data or 'height_plateaus' not in data:
            raise ValueError("Database must contain 'building_limits' and 'height_plateaus' keys.")

        # Normalize GeoJSON structure
        data = ensure_properties_in_geojson(data)

        # Prepare the final GeoJSON dictionary
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

        # Prepare the data to be written with the correct structure
        data = {
            'building_limits': {
                "type": "FeatureCollection",
                "features": building_limits_features
            },
            'height_plateaus': {
                "type": "FeatureCollection",
                "features": height_plateaus_features
            }
        }

        # Write data to Firebase
        ref.set(data)
        print("Polygons successfully written to Firebase.")
    except Exception as e:
        raise ValueError(f"Failed to write polygons to Firebase: {str(e)}")


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

        print(height_plateaus_gdf)


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
        height_plateaus_gdf['elevation'] = pd.to_numeric(
            height_plateaus_gdf['elevation'], errors='coerce'
        ).fillna(0.0)

        minx, miny, maxx, maxy = building_limits_gdf.total_bounds
        h_minx, h_miny, h_maxx, h_maxy = height_plateaus_gdf.total_bounds
        minx = min(minx, h_minx)
        miny = min(miny, h_miny)
        maxx = max(maxx, h_maxx)
        maxy = max(maxy, h_maxy)

        # Define resolution (adjust as needed)
        resolution = 0.000005  # Approx ~0.5 meters at the equator
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
            dtype='int16',
            all_touched=True
        )
        print("building_limits_raster")
        # building_limits_raster = building_limits_raster.astype(rasterio.float64)


        # Initialize height_plateaus_raster with zeros (float64 for higher precision)
        height_plateaus_raster = np.zeros((height, width))


        # Rasterize each height plateau and use maximum value in case of overlaps
        for idx, row in height_plateaus_gdf.iterrows():
            elevation = int(row.get('elevation', 0))
            geom = row.geometry
            if geom.is_empty:
                continue
            mask_raster = rasterio.features.rasterize(
                [(geom, 1)],  # Use 1 as the value for the mask
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='int16',
                all_touched=True
            )
            # raster = mask_raster.astype(rasterio.float64)  # Convert to float64 and apply elevation


            # Multiply the mask by elevation to apply the height
            raster = mask_raster * elevation
            print("multiplied elevation")
            height_plateaus_raster += raster
        print("building_limits_raster")

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
        height_da = height_da.where(building_mask, other=0)

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


def validate_height_plateaus_cover_building_limits(
    building_limits_gdf: gpd.GeoDataFrame,
    height_plateaus_gdf: gpd.GeoDataFrame
) -> Tuple[bool, np.ndarray]:
    """
    Validates whether all areas inside the building limits are covered by height plateaus.
    Returns:
        - A boolean indicating if the building limits are fully covered.
        - A NumPy array of coordinates where uncovered areas are found (if any).
    """
    try:
        # Define rasterization parameters
        minx, miny, maxx, maxy = building_limits_gdf.total_bounds

        # Define resolution (adjust as needed)
        resolution = 0.000005  # Approx ~0.5 meters at the equator
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
            dtype="int16",
            all_touched=True
        )

        # Rasterize height plateaus
        height_plateaus_raster = np.zeros((height, width), dtype="int16")
        for _, row in height_plateaus_gdf.iterrows():
            mask_raster = rasterio.features.rasterize(
                [(row.geometry, 1)],  # Use 1 as the value for the mask
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="int16",
                all_touched=True
            )
            height_plateaus_raster = np.maximum(height_plateaus_raster, mask_raster)

        # Check for uncovered areas inside building limits
        uncovered_mask = (building_limits_raster == 1) & (height_plateaus_raster == 0)

        # Get coordinates of uncovered areas
        uncovered_coords = np.argwhere(uncovered_mask)

        # Convert raster indices to geographic coordinates
        uncovered_geo_coords = [
            (minx + resolution * x, maxy - resolution * y) for y, x in uncovered_coords
        ]

        # Return validation result
        is_fully_covered = not uncovered_mask.any()
        return is_fully_covered, np.array(uncovered_geo_coords)

    except Exception as e:
        raise ValueError(f"Validation failed: {str(e)}")


def add_height_plateau_to_firebase(geometry: Dict, height: float):
    """
    Adds a new height plateau to the Firebase database, preserving the required structure.
    """
    try:
        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Ensure `height_plateaus` is structured correctly
        if "features" not in geo_dict["height_plateaus"]:
            geo_dict["height_plateaus"] = {"type": "FeatureCollection", "features": []}

        # Add the new plateau
        new_plateau = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {"elevation": height}  # Ensure 'properties' and 'elevation' keys are present
        }
        geo_dict["height_plateaus"]["features"].append(new_plateau)

        # Validate and normalize structure
        # validate_geojson_structure(geo_dict)

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


def delete_height_plateau_from_firebase(target_elevation: float):
    """
    Deletes a specific height plateau with a given elevation value from the Firebase database.
    
    Parameters:
    - target_elevation: The elevation value of the plateau to delete.
    """
    try:
        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Check if height_plateaus has the expected structure
        if "features" not in geo_dict["height_plateaus"]:
            print("No features found in height_plateaus.")
            return

        # Filter out the plateau with the target elevation
        filtered_features = [
            feature for feature in geo_dict["height_plateaus"]["features"]
            if feature["properties"].get("elevation") != target_elevation
        ]

        # Update the height_plateaus with the filtered features
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

        print(f"Height plateau with elevation {target_elevation} deleted successfully!")
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

        # Update the building limits to include the new geometry
        geo_dict["building_limits"] = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": new_geometry,
                    "properties": {}  # Ensure 'properties' key is present
                }
            ]
        }

        # Convert to GeoDataFrames
        building_limits_gdf = gpd.GeoDataFrame.from_features(
            geo_dict["building_limits"]["features"], crs="epsg:4326"
        )
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(
            geo_dict["height_plateaus"]["features"], crs="epsg:4326"
        )

        # Write back to Firebase
        write_polygons_to_firebase(building_limits_gdf, height_plateaus_gdf)

        print("Building limits updated successfully!")
    except Exception as e:
        print(f"Failed to modify building limits: {str(e)}")



def split_building_limits_by_height_plateaus():
    """
    Intersects building limits with height plateaus, splits the building limits
    based on the intersections, and writes the resulting geometries into Firebase
    under 'splitted_building_limits'.
    """
    try:
        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Convert to GeoDataFrames
        building_limits_gdf = gpd.GeoDataFrame.from_features(
            geo_dict["building_limits"]["features"], crs="epsg:4326"
        )
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(
            geo_dict["height_plateaus"]["features"], crs="epsg:4326"
        )

        # Ensure the GeoDataFrames are not empty
        if building_limits_gdf.empty or height_plateaus_gdf.empty:
            raise ValueError("Building limits or height plateaus are empty. Cannot proceed with splitting.")

        # Perform intersection
        splitted_geometries = []
        for _, building_row in building_limits_gdf.iterrows():
            for _, plateau_row in height_plateaus_gdf.iterrows():
                intersection = building_row.geometry.intersection(plateau_row.geometry)
                if not intersection.is_empty:
                    splitted_geometries.append({
                        "type": "Feature",
                        "geometry": mapping(intersection),
                        "properties": building_row.get("properties", {})  # Retain properties if any
                    })

        # Ensure the structure matches 'splitted_building_limits'
        splitted_building_limits = {
            "type": "FeatureCollection",
            "features": splitted_geometries
        }

        # Write the splitted building limits to Firebase
        ref = db.reference("/")
        ref.child("splitted_building_limits").set(splitted_building_limits)

        print("Splitted building limits written successfully!")
    except Exception as e:
        print(f"Failed to split building limits: {str(e)}")


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
        logging.info("rasterize_geodataframes completed")

        response_data = {
            "height_da": height_da.values.tolist(),  # Convert NumPy array to list
            "x_coords": height_da.coords["x"].values.tolist(),  # Serialize x-coordinates
            "y_coords": height_da.coords["y"].values.tolist(),  # Serialize y-coordinates
            "bounds": bounds,
        }
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rasterize: {str(e)}")


@app.post("/split-building-limits")
def split_building_limits():
    """Splits building limits by intersecting with height plateaus."""
    try:
        split_building_limits_by_height_plateaus()
        return {"message": "Splitted building limits written successfully to Firebase."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to split building limits: {str(e)}")

@app.get("/validate-height-plateaus")
def validate_height_plateaus():
    """Validate whether all areas inside building limits are covered by height plateaus."""
    try:
        geo_dict = read_geojson_from_firebase()
        building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)

        is_covered, uncovered_coords = validate_height_plateaus_cover_building_limits(
            building_limits_gdf, height_plateaus_gdf
        )

        return {
            "is_covered": is_covered,
            "uncovered_coords": uncovered_coords.tolist()  # Convert to list for JSON serialization
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# @app.get("/visualize")
# def visualize():
#     """Visualize the building limits and height plateaus."""
#     try:
#         geo_dict = read_geojson_from_firebase()
#         building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)
#         height_da, bounds, _, _ = rasterize_geodataframes(building_limits_gdf, height_plateaus_gdf)
#         visualize_height_da(height_da, building_limits_gdf, height_plateaus_gdf, bounds)
#         return {"message": "Visualization completed. Check your plot window."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to visualize: {str(e)}")

