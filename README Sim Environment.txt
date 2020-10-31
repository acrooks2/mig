README to Create New Simulation Environment

***Follow the steps.

***To create a simulation environment in an Area of Interest (AOI), use pre_process_basic.py

***It will require as input vector data representing administrative divisions at the level of granularity required for the simulation environment stored in the 'data' folder.

For example, the Central African Republic (CAR) Administrative Level 2 (prefectures, like provinces) shapefile will create a simulation environment of 16 location nodes
across the spatial extent of the country.
CAR Administrative Level 3 (subprefectures, like districts or counties) shapefile will create a simulation environment of 48 location nodes across the spatial extent of the country.
Less granular simulation environments will create less granular predictions but will run more quickly and require less computational overhead. 

***The shapefile will require at least four attributes:

1. Name or ID field containing the name or ID (string or integer) attributed to each individual location node. The name is specified in Line 28 of the script and referenced throughout.

2. Area field (a geometry field typical of all shapefiles)

3. Perimeter field (a geometry field typical of all shapefiles)
* The shapefile's geometry is called by geopandas throughout the script

4. REFPOP field containing the pre-existing refugee population for each administrative area at time of simulation initiation. 

Notes on REFPOP:

If no shapefile exists with the refugee population required for the simulation, consider a spatial join from an Excel file or other tabular data format containing relevant refugee statistics.
This can be accomplished using QGIS or ArcMap (using the field calculator), or natively in geopandas. 
If statistics for refugee populations only exist at one Administrative Level above the target level for the simulation environment (i.e. there are refugee statistics for Admnin Level 2
but Admin Level 3 is desired for the simulation), there is logic to support proxy calculations of refugee statistics at the required level available in the pre_process_conversion.py script.

***Open the pre_process_basic.py script and read in the shapefile from the location specified on your local machine in Line 15. 
***New shapefiles will be written out to the working directories specified in Lines 58 and 67. These shapefiles can be visualized using QGIS, ArcMap, or other geospatial visualization software.
***The simulation environment will display after the script runs as specified in Lines 89-91

