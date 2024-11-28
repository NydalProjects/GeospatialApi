# **Geospatial API - ReadMe**

This project provides a geospatial API for managing building limits and height plateaus, along with a test script (`run.py`) to interact with the API. The user only needs **Python 3.11** and the required dependencies to run the API functionalities.

---

## **Prerequisites**
- **Python 3.11/12 installed on your system
- Internet access for API requests

---

## **Setup Instructions**

### **1. Clone or Download the Project**
Ensure you have the `run.py` script in your working directory.

### **2. Install Dependencies**
Install the required packages using `pip`:

pip install requests geopandas matplotlib numpy xarray

### Running the Functions
All functionality is encapsulated in run.py. The user needs to edit and uncomment specific function calls to run the desired operations.

Open run.py in your favorite text editor.
Uncomment the desired function calls to execute the corresponding operations.


### Functionality Overview
1. Upload Shapes to Firebase
Upload pre-defined building limits and height plateaus to Firebase:

upload_shapes_to_firebase()

2. Update Firebase Data
Modify building limits and add height plateaus:
update_firebase()

3. Delete a Height Plateau by Elevation
Delete a specific height plateau by specifying its elevation value:
delete_height_plateau_by_elevation(3.75)

4. Rasterize Data
Fetch rasterized geospatial data and visualize it:
rasterized_data, bounds = rasterize()
if rasterized_data is not None and rasterized_data.size > 0:
    visualize_rasterized_data(rasterized_data, bounds)

5. Split Building Limits
Split building limits based on intersections with height plateaus:
split_building_limits()

6. Validate Height Plateaus
Check if building limits are fully covered by height plateaus:
validate_height_plateaus()
