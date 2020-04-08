"""
ABMs in the Study of Forced Migration
Author: Melonie Richey
Last Edited: 2020

Spatially-Explicit Agent-Based Model (ABM) of Forced Migration from Syria to Turkey ###
"""
import os
import time
import math
import copy
import random
import unidecode
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool

# CONSTANTS
# Percentage of refugees that move if in a district with one or more refugee camps
PERCENT_MOVE_AT_CAMP = 0.3
# Percentage of refugees that move if in a district with one of more conflict events
PERCENT_MOVE_AT_CONFLICT = 1
# Percentage of refugees that move if in a district without a conflict event or a camp
PERCENT_MOVE_AT_OTHER = 0.7
# Number of refugees that cross the Syrian-Turkish border at each time step
SEED_REFS = 0

NUM_FRIENDS = 1  # random.randint(1,3)
NUM_KIN = 1  # random.randint(1,3)

# Districts that contain open border crossings during month of simulation start
BORDER_CROSSING_LIST = [0, 1]  # ['Merkez Kilis', 'KarkamA+-A', 'YayladaAA+-', 'Kumlu']
# # Point to calculate western movement
LONDON_COORDS = (51.5074, -0.1278)

NUM_CHUNKS = 4  # mp.cpu_count()

DATA_DIR = './data'

MULTIPROCESS = True

NUM_STEPS = 1  # 1 step = 1 day

class Ref(object):
    """
    Class representative of a single refugee
    """
    def __init__(self, node):
        self.node = node  # Not used. Agent ID == Index in Sim.all_refugees
        self.kin_list = {}
        self.friend_list = {}


class Sim(object):
    """
    Class representative of the simulation
    """
    def __init__(self, graph, num_steps=10):
        self.graph = graph
        self.num_steps = num_steps
        self.max_pop = None
        self.num_refugees = sum([self.graph.nodes[n]['weight'] for n in self.graph.nodes])
        print('creating complete graph')
        self.all_refugees = nx.Graph()  # complete_graph(self.num_refugees)
        self.all_refugees.add_nodes_from([ref for ref in range(self.num_refugees)])
        self.kin_dict = {ref: set() for ref in self.all_refugees.nodes()}
        print(list(self.all_refugees.nodes)[:10])
        print('Assigning refugees to nodes')
        # assign refugees to nodes
        count = 0
        values = {}
        for node in self.graph.nodes():
            values.update({ref: node for ref in range(count, count+self.graph.nodes[node]['weight'])})
            count += self.graph.nodes[node]['weight']

        nx.set_node_attributes(self.all_refugees, values, 'location')
        # nx.set_node_attributes(self.all_refugees, set(), 'kin')
        nx.set_edge_attributes(self.all_refugees, 0, 'weight')

        print('Creating social links')
        # create social links between refs
        for ref in self.all_refugees.nodes():
            self.kin_dict[ref] = set()
            for x in range(NUM_KIN):
                kin = ref
                while kin == ref:  # (kin in self.all_refugees.nodes[index]['kin'])
                    kin = random.randint(0, self.num_refugees-1)
                self.kin_dict[ref].add(kin)
                # set for other kin
                self.kin_dict[kin].add(ref)
            # create friends
            for x in range(NUM_FRIENDS):
                friend = ref
                while friend == ref:
                    friend = random.randint(0, self.num_refugees-1)
                self.all_refugees.add_edge(ref, friend, weight=2.0)  # 7 = lowest friend. 9 is initializer.

    def find_new_node(self, node, ref):

        # find neighbor with highest weight
        neighbors = list(self.graph.neighbors(node))

        # initialize max node value to negative number
        most_desirable_score = -99
        most_desirable_neighbor = None

        # check to see if there are neighbors (in case node is isolate)
        if len(neighbors) == 0:
            # print(ref_pop, "refugees can't move from isolates", node)
            return

        kin_nodes = [self.all_refugees.nodes[kin]['location'] for kin in self.kin_dict[ref]]
        friend_nodes = [neigh for neigh in self.all_refugees.neighbors(ref)]  # if self.all_refugees.edges[ref, neigh]['weight'] >= 7

        # calculate neighbor with highest population
        for n in neighbors:
            refpop_norm = self.graph.nodes[n]['weight'] / self.max_pop
            refugees_at_node = self.graph.nodes[n]['weight']
            kin_at_node = kin_nodes.count(n)
            friends_at_node = friend_nodes.count(n)
            location_score = self.graph.nodes[n]['location_score']
            desirability = refpop_norm + kin_at_node + (friends_at_node / 2) + location_score
            # print(n, count_kin_at_node, count_friends_at_node, location_score, desirability)
            if desirability > most_desirable_score:
                most_desirable_score = desirability
                most_desirable_neighbor = n

        return most_desirable_neighbor

    def process_refs(self, refs):
        print('Starting process...')

        new_refs = {}
        new_weights = {key: 0 for key in self.graph.nodes}

        for ref in refs:
            node = self.all_refugees.nodes[ref]['location']
            num_conflicts = self.graph.nodes[node]['num_conflicts']
            num_camps = self.graph.nodes[node]['num_camps']

            if num_conflicts > 0:
                # Conflict zone - 1.0 move chance
                move = True
            elif num_camps:
                # At a camp - 0.003 move chance
                move = random.random() < PERCENT_MOVE_AT_CAMP
            else:
                # Neither camp nor conflict - 0.5 move chance
                move = random.random() < PERCENT_MOVE_AT_OTHER

            if not move:
                # new_refs[ref] = ref
                continue

            # Create a copy of refugee, assign new node attribute
            new_node = self.find_new_node(node, ref)
            if new_node is None:
                # new_refs.append(ref)
                continue
            new_refs[ref] = new_node

            # Update weight dict
            new_weights[node] -= 1
            new_weights[new_node] += 1

        # return new refugee list and node weight updates for these refs
        return new_refs, new_weights

    def step(self, step):
        nx.get_node_attributes(graph, 'weight')
        orig_weights = nx.get_node_attributes(graph, 'weight')

        self.max_pop = max(orig_weights.values())
        chunked_refs = np.array_split([x for x in range(len(self.all_refugees.nodes))], NUM_CHUNKS)

        if MULTIPROCESS:
            results = Pool(NUM_CHUNKS).map(self.process_refs, chunked_refs)
        else:
            results = []
            for chunk in chunked_refs:
                results.append(self.process_refs(chunk))

        new_refugee_nodes = {}
        new_weights = []
        new_weights.append(orig_weights)
        for result in results:
            new_refugee_nodes.update(result[0])
            new_weights.append(result[1])

        # self.all_refugees = new_refs

        nx.set_node_attributes(self.all_refugees, new_refugee_nodes, 'location')

        new_weights = pd.DataFrame(new_weights)
        new_weights = dict(zip(self.graph.nodes, list(new_weights.sum(numeric_only=True))))
        nx.set_node_attributes(self.graph, new_weights, 'weight')

        # seed border crossing nodes with new refugees
        start_index = new_ref_index = self.num_refugees
        self.num_refugees += SEED_REFS * len(BORDER_CROSSING_LIST)

        old_refugees = copy.copy(self.all_refugees)

        self.all_refugees = nx.complete_graph(self.num_refugees)
        self.all_refugees.update(old_refugees)

    def stop(self):
        for node in BORDER_CROSSING_LIST:
            self.graph.nodes[node]['weight'] += SEED_REFS
            for ref in range(new_ref_index, new_ref_index + SEED_REFS):
                self.kin_dict[ref] = set()
            new_ref_index += SEED_REFS

            # [self.all_refugees.add_node(ref+new_ref_index) for ref in range(0, SEED_REFS)]
            # values = [self.kin_dict[ref] = set() for ref in range(0, SEED_REFS)]
            # nx.set_node_attributes(self.all_refugees, values, '')
            # new_ref_index += SEED_REFS

        for index in range(start_index, len(self.all_refugees.nodes)):
            # create kin
            for x in range(NUM_KIN):
                kin = index
                while (kin == index):  # or (kin in self.all_refugees.nodes[index]['kin'])
                    kin = random.randint(0, self.num_refugees-1)
                self.kin_dict[index].add(kin)
                # set for other kin
                self.kin_dict[kin].add(index)

            # create friends
            for x in range(NUM_FRIENDS):
                friend = index
                while friend == index:
                    friend = random.randint(0, self.num_refugees-1)
                self.all_refugees.edges[index, friend]['weight'] = 9.0  # 7 = lowest friend. 9 is initializer.

    def run(self):
        # Add status bar for simulation run
        avg_step_time = 0
        for x in list(range(self.num_steps)):
            start = time.time()
            print(f'Starting step {x+1}')
            self.step(x)
            step_time = time.time() - start
            avg_step_time += step_time
            print(f'Step Time: {step_time:.2f}')
        print(f'Average step time: {step_time:.2f}')


def build_graph():
    # ***Data Engineering Using GeoPandas***

    # Get shapefile of Turkish districts (and re-project)
    polys = gpd.read_file(os.path.join(DATA_DIR, 'gadm36_TUR_2.shp'))

    # Check and set spatial projection
    # polys = polys.to_crs({'init': 'epsg:4326'})
    # print(polys.crs)

    # Get refugee population by province data (and re-project), from Turkish Statistical Institute, February 2019
    pop_by_province = gpd.read_file(os.path.join(DATA_DIR, 'REFPOP.shp'))

    # Check and set spatial projection
    # pop_by_province = pop_by_province.to_crs({'init': 'epsg:4326'})
    # print(pop_by_province.crs)

    # ***Create nodes***

    nodes = []
    # Remove non-english characters and fix duplicate district names ("Merkez")
    for index, row in polys.iterrows():
        name = unidecode.unidecode(row.NAME_2)
        if name in "Merkez":
            name += " " + unidecode.unidecode(row.NAME_1)
        if name in nodes:
            name += str(index)
        nodes.append(name)
        polys.at[index, "NAME_2"] = name
        pop_by_province.at[index, "NAME_2"] = name  # Update name in population file

    # ***Add Neighbors***

    # Add column to store list of neighbors
    polys["NEIGHBORS"] = None

    # Calculate neighbors and create an edge in the network between each pair of districts that share a border
    edges = []
    for index, row in polys.iterrows():
        neighbors = polys[polys.geometry.touches(row['geometry'])].NAME_2.tolist()
        for n in neighbors:
            edge = (row.NAME_2, n)
            edge = sorted(edge)
            edges.append(edge)
        polys.at[index, "NEIGHBORS"] = ", ".join(neighbors)

    # ***Add Population Data***

    # Create new column in target shapefile for refugee population
    polys["REFPOP"] = None

    # Take refugee population by province, divide by number of districts in province, assign each equivalent value as REFPOP of district
    for index, row in pop_by_province.iterrows():
        REFPOP_calc = row['REFPOP'] / row['count']
        # print(REFPOP_calc)
        if not math.isnan(REFPOP_calc):
            REFPOP_calc = int(REFPOP_calc)
        polys.REFPOP.iloc[[polys.NAME_1 == row.NAME_1]] = REFPOP_calc

    # Set REFPOP to 0 for undefined districts
    polys[['REFPOP']] = polys[['REFPOP']].fillna(value=0)

    # Write out new shapefile with neighbors and population attributes
    polys.to_file(os.path.join(DATA_DIR, "output_1.shp"))

    # ***Create Additional Spatial Data Layers***

    # Create Conflict data layer
    # Read in ACLED event data from February 2019 (and set projection)
    df_conflict = pd.read_csv(os.path.join(DATA_DIR, 'FebACLEDextract.csv'))
    conflict = gpd.GeoDataFrame(df_conflict, geometry=gpd.points_from_xy(df_conflict.longitude, df_conflict.latitude))
    conflict.crs = {'init': 'epsg:4326'}
    # print(conflict.crs)
    # print(conflict.head())

    # Plot events as points
    base = polys.plot(color='white', edgecolor='black')
    conflict.plot(ax=base, marker='o', color='red', markersize=5)

    # Create new column in target shapefile for count of conflict events per district
    polys["conflict"] = polys.apply(lambda row: sum(conflict.within(row.geometry)), axis=1)

    # print(polys.dtypes)
    # print(polys.conflict)

    # Write out test shapefile for verification in QGIS
    # polys.to_file("test.shp")

    # Plot conflict shapefile
    # colors = 6
    # figsize = (26, 20)
    # cmap = 'Blues'
    # conflict = polys.conflict
    # conflict.plot(column=conflict, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)
    # plt.show()

    # Read in refugee camp data (and re-project) from UNHCR Regional IM Working Group February 2019 (updated every 6 months)
    camps = gpd.read_file(os.path.join(DATA_DIR, 'tur_camps.shp'))
    # print(camps.crs)
    # print(camps)

    # Plot camps as points
    # base = polys.plot(color='white', edgecolor='black')
    # camps.plot(ax=base, marker='o', color='blue', markersize=5)
    # plt.show()

    # Create new column in target shapefile for refugee camps
    polys["camps"] = polys.apply(lambda row: sum(camps.within(row.geometry)), axis=1)

    # Write out test shapefile for verification in QGIS
    # polys.to_file("secondtest.shp")

    # ***Create Nodes from District Polygons***

    # Create centroids file (eventual nodes in the network)
    points = polys.copy()

    # Calculate centroids of polygons
    points['geometry'] = points['geometry'].centroid

    # Calculate location score. Districts closest to London scored highest.
    x1 = LONDON_COORDS[1]
    y1 = LONDON_COORDS[0]
    points['location_score'] = points.apply(lambda row: math.sqrt((row.geometry.x-x1)**2 + (row.geometry.y-y1)**2), axis=1)
    max_distance = max(list(points['location_score']))
    points['location_score'] = points.apply(lambda row: 1 - (row.location_score / max_distance), axis=1)

    # ***Network Creation using NetworkX***
    graph = nx.Graph()

    # ***Add Nodes to Graph***
    positions = {}
    for index, row in points.iterrows():
        node = row.NAME_2
        # Add the coordinates to the nodes so they can be displayed geospatially
        coords = row.geometry
        # weight_calc = row['REFPOP'] / polys['count']
        # weight = weight_calc
        weight = row.REFPOP

        location_score = row.location_score

        graph.add_node(node, pos=coords, weight=weight, num_conflicts=row.conflict, num_camps=row.camps,
                       location_score=location_score, incoming_refs=[], outgoing_refs=[])
        positions[node] = (coords.x, coords.y)

    # Write centroids to new Shapefile
    points.to_file(os.path.join(DATA_DIR, 'output_2_centroids.shp'))

    # ***Add Edges to Graph***
    for edge in edges:
        graph.add_edge(edge[0], edge[1], weight=1)

    # ***Draw Graph***
    # base = polys.plot(color='cadetblue', edgecolor='black')
    # nx.draw(graph, node_size=25, node_color='darkblue', pos=positions)
    # plt.show()

    return graph, polys


if __name__ == '__main__':
    """
    Program Execution starts here
    """

    print('Building input files')
    # graph, polys = build_graph()

    graph = nx.complete_graph(50)
    nx.set_node_attributes(graph, name='weight', values=500)
    nx.set_node_attributes(graph, name='num_conflicts', values=0)
    nx.set_node_attributes(graph, name='num_camps', values=1)
    nx.set_node_attributes(graph, name='location_score', values=0.5)

    # Run Sim
    start = time.time()
    sim = Sim(graph, NUM_STEPS)
    print('Time to build -', time.time() - start)


    start_node_weights = nx.get_node_attributes(graph, 'weight')
    sim.run()
    end_node_weights = nx.get_node_attributes(graph, 'weight')

    # print starting and ending node weights
    # for node in graph.nodes:
    #     print(node, start_node_weights[node], end_node_weights[node])

    print("Total start weight:", sum(start_node_weights.values()))
    print("Total end weight:", sum(end_node_weights.values()))


def stop():
    # Write end_node_weights to final_REFPOP column in shapefile
    output = gpd.read_file(os.path.join(DATA_DIR, 'output_1.shp'))

    output['simEnd'] = output['NAME_1'].map(end_node_weights)
    # output['simEnd'] = end_node_weights.values()

    # write out to shapefile
    output.to_file(os.path.join(DATA_DIR, 'output_3_simOutput.shp'))

    # visualize simulation output
    #output = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\Summer 2019 Comp Migration\output_3_simOutput.shp")
    #colors = 6
    #figsize = (26, 20)
    #cmap = 'winter_r'
    #simEnd = output.simEnd
    #output.plot(column=simEnd, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)

    ## MODEL VALIDATION ##
    val = gpd.read_file(os.path.join(DATA_DIR, 'gadm36_TUR_1_val.shp'))
    polys = gpd.read_file(os.path.join(DATA_DIR, 'REFPOP.shp'))
    # sim_val = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\Summer 2019 Comp Migration\output_3_simOutput.shp")

    # Remove non-english characters and fix duplicate district names ("Merkez")
    for index, row in val.iterrows():
        name = unidecode.unidecode(row.NAME_1)
        val.at[index, "NAME_1"] = name

    # Take refugee population by province, divide by number of districts in province, assign each equivalent value as REFPOP of district
    for index, row in val.iterrows():
        # print(type(row['val_mar19']))
        # print(type(polys_val.loc[index].count))
        val_calc = row['val_mar19'] / float(polys.loc[index]['count'])
        if not math.isnan(val_calc):
            val_calc = int(val_calc)
        polys.REFPOP.iloc[[polys.NAME_1 == row.NAME_1]] = val_calc

    # Write out new shapefile with refugee validation population by district
    val.to_file(os.path.join(DATA_DIR, 'output_4.shp'))


    # C reate new colums for normalized values and comparison
    val["val_mar19_norm"] = None
    output["simEnd_norm"] = None
    output["accuracy"] = None

    # Normalize both actual and predicted REFPOP for district-level comparison
    print(max(val.val_mar19))
    print(min(val.val_mar19))
    val['val_mar19_norm'] = ((val.val_mar19) - min((val.val_mar19))) / (max((val.val_mar19)) - min((val.val_mar19)))
    print(val.val_mar19_norm)

    print(max(output.simEnd))
    print(min(output.simEnd))
    output['simEnd_norm'] = ((output.simEnd) - min(output.simEnd)) / ((max(output.simEnd)) - min(output.simEnd))
    print(output.simEnd_norm)

    # Comparative scaled_actual & scale_predicted
    output['accuracy'] = (output.simEnd_norm - val.val_mar19_norm)

    # Write out new shapefile with validation accuracy by district
    output.to_file(os.path.join(DATA_DIR, 'output_5_validation.shp'))

    # visualize validation output - unnecessary to read again
    validation = gpd.read_file(os.path.join(DATA_DIR, 'output_5_validation.shp'))
    colors = 6
    figsize = (26, 20)
    cmap = 'winter_r'
    accuracy = validation.accuracy
    validation.plot(column=accuracy, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)

    # SHOW PLOTS
    plt.show()
