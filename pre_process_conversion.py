import random
import unidecode
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import math
import matplotlib.pyplot as plt

# Get shapefile of desired simulation environment
polys = gpd.read_file(r"C:\Users\mrich\Downloads\mig-master - Copy for Africa\mig-master\data\CAR_2_pop.shp")
#print(polys)

# Get refugee population by province data, from HDX/UNOCHA, June 2019
pop_by_province = gpd.read_file(r"C:\Users\mrich\Downloads\mig-master - Copy for Africa\mig-master\data\CAR_REFPOP.shp")

nodes = []
# Remove non-english characters in desired name row
# Replace desired name row with column header in imported shapefile
# Here it is "admin2"
print("Simulation Environment Locations")
for index, row in polys.iterrows():
    name = unidecode.unidecode(row.admin2)
    if name in nodes:
        name += str(index)
    nodes.append(name)
    polys.at[index, "admin2"] = name
    pop_by_province.at[index, "admin2"] = name  # Update name in population file
    print(name)

## ADD NEIGHBORS ##

# Add column to store list of neighbors
polys["NEIGHBORS"] = None

# Calculate neighbors and create an edge in the network between each pair of districts that share a border
edges = []
for index, row in polys.iterrows():
    neighbors = polys[polys.geometry.touches(row['geometry'])].admin2.tolist()
    for n in neighbors:
        edge = (row.admin2, n)
        edge = sorted(edge)
        edges.append(edge)
    polys.at[index, "NEIGHBORS"] = ", ".join(neighbors)

## ADD POPULATION DATA ##

# Create new column in target shapefile for refugee population
polys["REFPOP"] = None

# Get refugee population in each administrative area
# If refugee population is provided at one administrative level above that of the simulation environment,
# take refugee population by province, divide by number of districts in province, assign each equivalent value as REFPOP of district
# Note: requires column 'count'
# Change polys.XYZ to location names column header
print("Simulation Environment Locations with Pre-Existing Refugee Populations")
for index, row in pop_by_province.iterrows():
    REFPOP_calc = row['REFPOP'] # / row['count']
    names = row['admin2']
    print(names, REFPOP_calc)
    if not math.isnan(REFPOP_calc):
        REFPOP_calc = int(REFPOP_calc)
    polys.REFPOP.iloc[[polys.admin2 == row.admin2]] = REFPOP_calc

# Set REFPOP to 0 for undefined districts
polys[['REFPOP']] = polys[['REFPOP']].fillna(value=0)

## WRITE NEW SHAPEFILE ##

# Write out new shapefile with neighbors and population attributes
polys.to_file("./data/polys_w_neighbors.shp")

## CREATE NODES FROM DISTRICT POLYGONS ##

# Create centroids file (eventual nodes in the network)
points = polys.copy()
# Calculate centroids of polygons
points['geometry'] = points['geometry'].centroid
# Write centroids to new Shapefile
points.to_file("./data/centroids.shp")

##### NETWORK CREATION USING NETWORKX #####

## CREATE NETWORK ##
graph = nx.Graph()

# Add the coordinates to the nodes so they can be displayed geospatially
positions = {}
for index, row in points.iterrows():
    node = row.admin2
    coords = row.geometry
    weight = row.REFPOP
    graph.add_node(node, pos=coords, weight=weight)
    positions[node] = (coords.x, coords.y)

## Create graph ##
for edge in edges:
    graph.add_edge(edge[0], edge[1], weight=random.uniform(0, 1))

## Draw graph ##

base = polys.plot(color='skyblue', edgecolor='black')
nx.draw(graph, node_size=25, node_color='darkblue', pos=positions)
plt.show()