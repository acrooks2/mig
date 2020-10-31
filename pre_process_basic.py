import random
import unidecode
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import math
import matplotlib.pyplot as plt

# Get shapefile of desired simulation environment
# Shapefile should contain:
# 1. name column (here "admin2") for each distinct location
# 2. REFPOP column with pre-existing refugee population (if any)
# Note: consider a spatial join, manual creation of this column, or other operations if no shapefile exists with these data
polys = gpd.read_file(r"C:\Users\mrich\Downloads\mig-master - Copy for Africa\mig-master\data\CAR_3.shp")
#print(polys)

nodes = []
# Remove non-english characters in desired name row
# Replace desired name row with column header in imported shapefile
# Here it is "admin2"
# print("Simulation Environment Locations")
for index, row in polys.iterrows():
    name = unidecode.unidecode(row.admin2)
    if name in nodes:
        name += str(index)
    nodes.append(name)
    polys.at[index, "admin2"] = name
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

print("Simulation Environment Locations Key")
polys["REFPOP"] = row['REFPOP']
REFPOP = polys["REFPOP"]
# Set REFPOP to 0 for undefined districts
polys[['REFPOP']] = polys[['REFPOP']].fillna(value=0)

print("Simulation Environment Locations and REFPOP")
print(polys['admin2'], REFPOP)

## WRITE NEW SHAPEFILE ##

# Write out new shapefile with neighbors and population attributes
#polys.to_file("./data/polys_w_neighbors.shp")

## CREATE NETWORK LOCATION NODES FROM POLYGONS ##

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