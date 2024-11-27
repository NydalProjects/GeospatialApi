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
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import os
from google.cloud import secretmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import geopandas as gpd


def get_firebase_credentials():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/861222091615/secrets/firebase-secret/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")


def initialize_firebase():
    try:
        credentials_json = get_firebase_credentials()

        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(credentials_json))
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://nydalprojects-default-rtdb.europe-west1.firebasedatabase.app/'
            })
    except Exception as e:
        raise ValueError(f"Failed to initialize Firebase: {str(e)}")


app = FastAPI()

def ensure_properties_in_geojson(geo_dict: Dict) -> Dict:
    try:
        if 'features' in geo_dict.get('building_limits', {}):
            for feature in geo_dict['building_limits']['features']:
                if 'properties' not in feature:
                    feature['properties'] = {}

        if 'features' in geo_dict.get('height_plateaus', {}):
            for feature in geo_dict['height_plateaus']['features']:
                if 'properties' not in feature:
                    feature['properties'] = {}

                if isinstance(feature['properties'], dict) and 'properties' in feature['properties']:
                    feature['properties'].update(feature['properties'].pop('properties'))

                elevation = feature['properties'].get('elevation')
                feature['properties']['elevation'] = float(elevation) if elevation is not None else 0.0


    except KeyError as e:
        raise ValueError(f"Error in GeoJSON structure: {str(e)}")

    return geo_dict


def read_geojson_from_firebase() -> Dict:

    initialize_firebase()
    try:
        ref = db.reference('/')

        def read_transaction(current_data):
            if current_data is None:
                raise ValueError("Database is empty.")
            return current_data

        data = ref.transaction(read_transaction)

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
        initialize_firebase()

        building_limits_features = json.loads(building_limits_gdf.to_json())['features']
        height_plateaus_features = json.loads(height_plateaus_gdf.to_json())['features']

        ref = db.reference('/')

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

        def write_transaction(current_data):
            return data

        ref.transaction(write_transaction)

        print("Polygons successfully written to Firebase.")
    except Exception as e:
        raise ValueError(f"Failed to write polygons to Firebase: {str(e)}")



def read_geojson_create_geodataframe(geo_dict: Dict) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    try:
        
        building_limits_gdf = gpd.GeoDataFrame.from_features(
            geo_dict['building_limits'],
            crs='epsg:4326'
        )

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
        height_plateaus_gdf['elevation'] = pd.to_numeric(
            height_plateaus_gdf['elevation'], errors='coerce'
        ).fillna(0.0)

        minx, miny, maxx, maxy = building_limits_gdf.total_bounds
        h_minx, h_miny, h_maxx, h_maxy = height_plateaus_gdf.total_bounds
        minx = min(minx, h_minx)
        miny = min(miny, h_miny)
        maxx = max(maxx, h_maxx)
        maxy = max(maxy, h_maxy)

        resolution = 0.000005
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        total_cells = width * height
        if total_cells > 1e8:
            print(f"Warning: The raster size is very large ({total_cells} cells).")
            print("This may consume a lot of memory and take a long time to process.")

        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        building_limits_raster = rasterio.features.rasterize(
            ((geom, 1) for geom in building_limits_gdf.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='int16',
            all_touched=True
        )
        print("building_limits_raster")

        height_plateaus_raster = np.zeros((height, width))

        for idx, row in height_plateaus_gdf.iterrows():
            elevation = int(row.get('elevation', 0))
            geom = row.geometry
            if geom.is_empty:
                continue
            mask_raster = rasterio.features.rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='int16',
                all_touched=True
            )

            raster = mask_raster * elevation
            print("multiplied elevation")
            height_plateaus_raster += raster
        print("building_limits_raster")

        x_coords = np.linspace(minx, maxx, width)
        y_coords = np.linspace(maxy, miny, height)

        height_da = xr.DataArray(
            height_plateaus_raster,
            coords=[y_coords, x_coords],
            dims=["y", "x"]
        )
        building_mask = building_limits_raster == 1

        height_da = height_da.where(building_mask, other=0)

        inside_building_no_plateau = (building_mask) & (np.isnan(height_da))
        height_da = height_da.where(~inside_building_no_plateau, other=0)

        raster_crs = rasterio.crs.CRS.from_epsg(4326)
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

    try:

        minx, miny, maxx, maxy = building_limits_gdf.total_bounds

        resolution = 0.000005
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        total_cells = width * height
        if total_cells > 1e8:
            print(f"Warning: The raster size is very large ({total_cells} cells).")
            print("This may consume a lot of memory and take a long time to process.")

        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        building_limits_raster = rasterio.features.rasterize(
            ((geom, 1) for geom in building_limits_gdf.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="int16",
            all_touched=True
        )

        height_plateaus_raster = np.zeros((height, width), dtype="int16")
        for _, row in height_plateaus_gdf.iterrows():
            mask_raster = rasterio.features.rasterize(
                [(row.geometry, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="int16",
                all_touched=True
            )
            height_plateaus_raster = np.maximum(height_plateaus_raster, mask_raster)

        uncovered_mask = (building_limits_raster == 1) & (height_plateaus_raster == 0)

        uncovered_coords = np.argwhere(uncovered_mask)

        uncovered_geo_coords = [
            (minx + resolution * x, maxy - resolution * y) for y, x in uncovered_coords
        ]

        is_fully_covered = not uncovered_mask.any()
        return is_fully_covered, np.array(uncovered_geo_coords)

    except Exception as e:
        raise ValueError(f"Validation failed: {str(e)}")


def add_height_plateau_to_firebase(geometry: Dict, height: float):
    try:
        initialize_firebase()

        geo_dict = read_geojson_from_firebase()

        if "features" not in geo_dict["height_plateaus"]:
            geo_dict["height_plateaus"] = {"type": "FeatureCollection", "features": []}

        new_plateau = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {"elevation": height}
        }
        geo_dict["height_plateaus"]["features"].append(new_plateau)

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

    try:
        initialize_firebase()

        # Fetch existing data
        geo_dict = read_geojson_from_firebase()

        # Check if height_plateaus has the expected structure
        if "features" not in geo_dict["height_plateaus"]:
            print("No features found in height_plateaus.")
            return

        filtered_features = [
            feature for feature in geo_dict["height_plateaus"]["features"]
            if feature["properties"].get("elevation") != target_elevation
        ]

        geo_dict["height_plateaus"]["features"] = filtered_features

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

    try:
        initialize_firebase()

        geo_dict = read_geojson_from_firebase()

        geo_dict["building_limits"] = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": new_geometry,
                    "properties": {}
                }
            ]
        }

        building_limits_gdf = gpd.GeoDataFrame.from_features(
            geo_dict["building_limits"]["features"], crs="epsg:4326"
        )
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(
            geo_dict["height_plateaus"]["features"], crs="epsg:4326"
        )

        write_polygons_to_firebase(building_limits_gdf, height_plateaus_gdf)

        print("Building limits updated successfully!")
    except Exception as e:
        print(f"Failed to modify building limits: {str(e)}")



def split_building_limits_by_height_plateaus():

    try:
        initialize_firebase()

        geo_dict = read_geojson_from_firebase()

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

        splitted_building_limits = {
            "type": "FeatureCollection",
            "features": splitted_geometries
        }

        ref = db.reference("/")
        ref.child("splitted_building_limits").set(splitted_building_limits)

        print("Splitted building limits written successfully!")
    except Exception as e:
        print(f"Failed to split building limits: {str(e)}")


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


@app.get("/upload-shapes")
def upload_shapes_to_firebase():
    """
    Upload the `shapes.json` file to Firebase. The file must be located in the same directory as `main.py`.
    Ensures compatibility with Google Cloud Run deployments.
    """
    try:
        initialize_firebase()

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes.json")

        with open(file_path, 'r') as f:
            data = json.load(f)

        if "building_limits" not in data or "height_plateaus" not in data:
            raise ValueError("JSON file must contain 'building_limits' and 'height_plateaus' keys.")

        ref = db.reference('/')

        ref.set(data)
        return {"message": "shapes.json uploaded successfully to Firebase."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload shapes.json: {str(e)}")


@app.get("/read")
def read_data():
    try:

        geo_dict = read_geojson_from_firebase()
        return geo_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data: {str(e)}")


@app.post("/add-height-plateau")
def add_height_plateau(request: AddHeightPlateauRequest):
    try:
        add_height_plateau_to_firebase(request.geometry, request.height)
        return {"message": "Height plateau added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add height plateau: {str(e)}")


@app.delete("/delete-height-plateau")
def delete_height_plateau(request: DeleteHeightPlateauRequest):
    try:
        delete_height_plateau_from_firebase(request.height)
        return {"message": f"Height plateau with height {request.height} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete height plateau: {str(e)}")


@app.put("/modify-building-limits")
def modify_building_limits(request: ModifyRequest):
    try:
        modify_building_limits_in_firebase(request.new_geometry)
        return {"message": "Building limits updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify building limits: {str(e)}")


@app.put("/rasterize")
def rasterize():
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
    try:
        split_building_limits_by_height_plateaus()
        return {"message": "Splitted building limits written successfully to Firebase."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to split building limits: {str(e)}")

@app.get("/validate-height-plateaus")
def validate_height_plateaus():
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


