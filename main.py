from typing import Dict, Tuple, List, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import geopandas as gpd
import rasterio.features
import numpy as np
import xarray as xr
from shapely.geometry import Polygon, mapping
import traceback

app = FastAPI()

@app.exception_handler(Exception)
async def handle_error(request: Request, exc: Exception):
    tb = traceback.format_exc()
    error_message = f"An error occurred: {str(exc)}\n\n{tb}"
    return JSONResponse(status_code=500, content={"error": error_message})

@app.exception_handler(403)
async def forbidden(request: Request, exc: HTTPException):
    error_message = "Sorry, you do not have permission to access this resource."
    return JSONResponse(status_code=403, content={"error": error_message})

@app.exception_handler(404)
async def page_not_found(request: Request, exc: HTTPException):
    error_message = "Sorry, the requested page could not be found."
    return JSONResponse(status_code=404, content={"error": error_message})

def check_height_plateau_within_building_limits(rasters: Dict[str, np.ndarray]) -> bool:
    if not isinstance(rasters, dict):
        raise TypeError("rasters must be a dictionary of numpy arrays")
    
    building_limits = rasters.get("0.0")
    if building_limits is None or not isinstance(building_limits, np.ndarray):
        raise ValueError("Building limits must be a valid numpy array")
    
    for i, raster_row in rasters.items():
        if not isinstance(raster_row, np.ndarray):
            raise ValueError(f"Raster row {i} must be a valid numpy array")
        
        if float(i) <= 0.0:
            continue
        
        intersection = raster_row * building_limits
        intersection_sum = intersection.sum()
        xr1_sum = raster_row.sum()
        if not np.isclose(intersection_sum, xr1_sum, rtol=1e-03):
            return False
    
    return True

def cut_height_plateau_building_limits(rasters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    try:
        building_limits = rasters["0.0"]
    except KeyError:
        raise ValueError("Input dictionary must contain a key '0.0' with the building limits.")
        
    for i, raster_row in rasters.items():
        if i != "0.0":
            if not isinstance(i, float):
                raise TypeError("Keys in input dictionary other than '0.0' must be floats representing height plateau values.")
            raster_row[building_limits == 0] = 0
            rasters[str(i)] = raster_row

    return rasters

def find_holes(rasters: Dict[str, np.ndarray]) -> bool:
    rasters_maximum = None

    # Find the maximum height for each cell
    for i, raster_row in rasters.items():
        if i != "0.0":
            if rasters_maximum is None:
                rasters_maximum = raster_row
            else:
                rasters_maximum = np.maximum(rasters_maximum, raster_row)

    visited = rasters_maximum.copy().astype(bool)
    
    # Start a DFS from each boundary cell and mark all reachable cells as visited
    for i in range(rasters_maximum.shape[0]):
        for j in range(rasters_maximum.shape[1]):
            if i == 0 or j == 0 or i == rasters_maximum.shape[0]-1 or j == rasters_maximum.shape[1]-1:
                if not visited[i, j] and rasters_maximum[i, j] == 0:
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        visited[x, y] = True
                        if x > 0 and not visited[x-1, y] and rasters_maximum[x-1, y] == 0:
                            stack.append((x-1, y))
                        if x < rasters_maximum.shape[0]-1 and not visited[x+1, y] and rasters_maximum[x+1, y] == 0:
                            stack.append((x+1, y))
                        if y > 0 and not visited[x, y-1] and rasters_maximum[x, y-1] == 0:
                            stack.append((x, y-1))
                        if y < rasters_maximum.shape[1]-1 and not visited[x, y+1] and rasters_maximum[x, y+1] == 0:
                            stack.append((x, y+1))
    
    # Check if there are any 0 values that were not marked as visited
    if np.any(np.logical_and(rasters_maximum == 0, ~visited)):
        return True
    else:
        return False

def check_height_plateau_cover_building_limits(rasters: Dict[str, np.ndarray]) -> bool:
    try:
        maximum_height_plateaus = None

        for i, raster_row in rasters.items():
            if float(i) > 0.0:
                if maximum_height_plateaus is None:
                    maximum_height_plateaus = raster_row
                else:
                    maximum_height_plateaus = np.maximum(maximum_height_plateaus, raster_row)
        if np.array_equal(maximum_height_plateaus, rasters["0.0"]):
            return True
        else:
            return False
    except Exception as e:
        print("Error: Failed to check height plateau coverage of building limits")
        print(str(e))
        return False

def read_geojson_create_geodataframe(geo_dict: Dict) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    try:
        # Create GeoDataFrame from building_limits feature
        building_limits_gdf = gpd.GeoDataFrame.from_features(
            geo_dict['building_limits'],
            crs='epsg:4326'
        )

        # Create GeoDataFrame from height_plateaus feature and set "elevation" as index
        height_plateaus_gdf = gpd.GeoDataFrame.from_features(
            geo_dict['height_plateaus'],
            crs='epsg:4326'
        ).set_index('elevation')
    except KeyError as e:
        raise ValueError("Failed to create GeoDataFrame: {}".format(str(e)))

    return building_limits_gdf, height_plateaus_gdf

def split_building_limits(building_limits_gdf: gpd.GeoDataFrame, height_plateaus_gdf: gpd.GeoDataFrame) -> Union[gpd.GeoDataFrame, None]:
    try:
        # Get the intersection of the two GeoDataFrames
        intersect = gpd.overlay(building_limits_gdf, height_plateaus_gdf, how='intersection')

        # Drop any columns from height_plateaus_gdf that are not in the intersection
        cols = [c for c in height_plateaus_gdf.columns if c in intersect.columns]
        cut_geoms = height_plateaus_gdf[cols].intersection(building_limits_gdf.unary_union)

        # Create a new GeoDataFrame with the cut geometries
        cut_gdf = gpd.GeoDataFrame(geometry=cut_geoms)
        
        return cut_gdf

    except ValueError as e:
        raise ValueError("Failed to split building limits: {}".format(str(e)))

def rasterize_geodataframe(geodataframe: gpd.GeoDataFrame) -> Tuple[Dict[str, np.ndarray], float, float, float, float, float, float]:
    try:
        xmin, ymin, xmax, ymax = geodataframe.total_bounds
        xres, yres = 0.00001, 0.00001
        rasters = {}
        
        # Loop over each row in the GeoDataFrame and rasterize it
        for i, row in geodataframe.iterrows():
            if row.geometry is None:
                raise ValueError(f"Null geometry found in row {i}")
                
            if not isinstance(row.geometry, Polygon):
                raise ValueError(f"Geometry in row {i} is not a polygon")
            
            raster = rasterio.features.rasterize(
                [row.geometry],
                out_shape=(
                    int((ymax - ymin) / yres),
                    int((xmax - xmin) / xres)
                ),
                transform=rasterio.transform.from_bounds(
                    xmin, ymin, xmax, ymax,
                    int((xmax - xmin) / xres),
                    int((ymax - ymin) / yres)
                ),
                all_touched=True,
                fill=0,
                default_value=1,
                dtype=rasterio.uint8
            )
            
            rasters[str(i)] = raster

        return rasters, xmin, ymin, xmax, ymax, xres, yres
    
    except Exception as e:
        raise e

@app.post("/main")
async def main_endpoint(geo_json: Dict):
    try:
        geo_dict = dict(geo_json)

        building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(geo_dict)
        split_building_limits_gdf = split_building_limits(building_limits_gdf, height_plateaus_gdf)
        building_limits_height_plateaus_gdf = height_plateaus_gdf.append(building_limits_gdf)
        
        # Convert the geometry column to a dictionary using the __geo_interface__ attribute
        split_building_limits_gdf['geometry'] = split_building_limits_gdf['geometry'].apply(
            lambda x: mapping(x) if x else None
        )
        
        # Convert the GeoDataFrame to a JSON-serializable dictionary
        split_building_limits_gdf_dict = split_building_limits_gdf.to_dict(orient='records')

        building_limits_height_plateaus_rasters, xmin, ymin, xmax, ymax, xres, yres = rasterize_geodataframe(building_limits_height_plateaus_gdf)

        height_plateau_cover_building_limits = check_height_plateau_cover_building_limits(
            rasters=building_limits_height_plateaus_rasters
        )
        is_hole_in_height_plateau = find_holes(rasters=building_limits_height_plateaus_rasters)
        height_plateau_within_building_limits = check_height_plateau_within_building_limits(
            rasters=building_limits_height_plateaus_rasters
        )

        return {
            "height_plateau_within_building_limits": str(height_plateau_within_building_limits),
            "is_hole_in_height_plateau": str(is_hole_in_height_plateau),
            "height_plateau_cover_building_limits": str(height_plateau_cover_building_limits),
            "split_building_limits": split_building_limits_gdf_dict
        }

    except Exception as e:
        tb = traceback.format_exc()
        error_message = f"An error occurred: {str(e)}\n\n{tb}"
        return JSONResponse(status_code=500, content={"error": error_message})

@app.get("/health/readiness")
async def readiness():
    return 'App is ready'

@app.get("/health/liveness")
async def liveness():
    return 'App is living'
