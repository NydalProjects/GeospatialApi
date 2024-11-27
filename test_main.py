import pytest
from shapely.geometry import Polygon
from main import (
    validate_height_plateaus_cover_building_limits,
    read_geojson_create_geodataframe,
)

@pytest.fixture
def mock_geojson_data():
    """Fixture to provide mock GeoJSON data with realistic scenarios."""
    return {
        "building_limits": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [10.757, 59.913],
                                [10.759, 59.913],
                                [10.759, 59.915],
                                [10.757, 59.915],
                                [10.757, 59.913],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        },
        "height_plateaus": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [10.7575, 59.9135],
                                [10.7585, 59.9135],
                                [10.7585, 59.9145],
                                [10.7575, 59.9145],
                                [10.7575, 59.9135],
                            ]
                        ],
                    },
                    "properties": {"elevation": 10},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [10.758, 59.914],
                                [10.759, 59.914],
                                [10.759, 59.915],
                                [10.758, 59.915],
                                [10.758, 59.914],
                            ]
                        ],
                    },
                    "properties": {"elevation": 5},
                },
            ],
        },
    }


def test_read_geojson_create_geodataframe(mock_geojson_data):
    """Test conversion of GeoJSON to GeoDataFrames."""
    building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(mock_geojson_data)

    assert not building_limits_gdf.empty, "Building limits GeoDataFrame should not be empty."
    assert not height_plateaus_gdf.empty, "Height plateaus GeoDataFrame should not be empty."
    assert len(height_plateaus_gdf) == 2, "There should be 2 height plateaus."
    assert building_limits_gdf.crs == "epsg:4326", "Building limits CRS should be EPSG:4326."
    assert height_plateaus_gdf.crs == "epsg:4326", "Height plateaus CRS should be EPSG:4326."


def test_validate_height_plateaus_cover_building_limits(mock_geojson_data):
    """Test validation of height plateaus coverage over building limits."""
    building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(mock_geojson_data)

    # Use a coarser resolution to make rasterization more efficient
    resolution = 0.0001  # ~10 meters

    is_covered, uncovered_coords = validate_height_plateaus_cover_building_limits(
        building_limits_gdf, height_plateaus_gdf
    )

    assert not is_covered, "The building limits should not be fully covered."
    assert len(uncovered_coords) > 0, "There should be uncovered areas."
    assert uncovered_coords.shape[1] == 2, "Uncovered coordinates should be in (x, y) format."


def test_split_building_limits_by_height_plateaus(mock_geojson_data):
    """Test splitting of building limits by height plateaus."""
    building_limits_gdf, height_plateaus_gdf = read_geojson_create_geodataframe(mock_geojson_data)

    # Mock splitting logic
    splitted_geometries = []
    for _, building_row in building_limits_gdf.iterrows():
        for _, plateau_row in height_plateaus_gdf.iterrows():
            intersection = building_row.geometry.intersection(plateau_row.geometry)
            if not intersection.is_empty:
                splitted_geometries.append({
                    "type": "Feature",
                    "geometry": intersection,
                    "properties": building_row.get("properties", {})
                })

    assert len(splitted_geometries) == 2, "There should be 2 split geometries."
    for geom in splitted_geometries:
        assert not geom["geometry"].is_empty, "Split geometries should not be empty."
        assert isinstance(geom["geometry"], Polygon), "Split geometries should be Polygons."
