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
SEED_REFS = 10

NUM_FRIENDS = 1  # random.randint(1,3)
NUM_KIN = 1  # random.randint(1,3)

# Districts that contain open border crossings during month of simulation start
BORDER_CROSSING_LIST = []  # ['Merkez Kilis', 'KarkamA+-A', 'YayladaAA+-', 'Kumlu']
# # Point to calculate western movement
LONDON_COORDS = (51.5074, -0.1278)

NUM_CHUNKS = 1  # mp.cpu_count()

DATA_DIR = '../data'


class Ref(object):
    """
    Class representative of a single refugee
    """
    def __init__(self, node, num_refugees):
        self.node = node  # Not used. Agent ID == Index in Sim.all_refugees
        self.kin_list = {}
        self.friend_list = {}

    def create_social_links(self, index, sim):
        # create kin
        for x in range(NUM_KIN):
            kin = index
            while kin != index:
                kin = random.randint(0, sim.num_refugees)
            # establish kin list
            self.kin_list[kin] = 1
            # make links reciprocal
            sim.all_refugees[kin].kin_list[index] = 1

        # create friends
        for x in range(NUM_FRIENDS):
            friend = index
            while friend != index:
                friend = random.randint(0, sim.num_refugees)
            # establish friend list
            self.friend_list[friend] = 1
            # make links reciprocal
            sim.all_refugees[friend].friend_list[index] = 1


class Sim(object):
    """
    Class representative of the simulation
    """
    def __init__(self, graph, num_steps=10):
        self.graph = graph
        self.num_steps = num_steps
        # node weight = # of refugees at initialization
        self.num_refugees = sum([self.graph.nodes[n]['weight'] for n in self.graph.nodes])
        # all_refugees is a list that holds reference to each  refugee in network
        self.all_refugees = []
        for node in self.graph.nodes():
            self.all_refugees.extend([Ref(node, self.num_refugees) for x in range(self.graph.nodes[node]['weight'])])
        # create social friendship and kinsdhip ties
        for index, ref in enumerate(self.all_refugees):
            ref.create_social_links(index, self)
        self.max_pop = None

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

        # for each kin, get node location
        kin_nodes = [self.all_refugees[kin].node for kin in self.all_refugees[ref].kin_list]
        # for each friend, get node location
        friend_nodes = [self.all_refugees[friend].node for friend in self.all_refugees[ref].friend_list]

        # calculate neighbor with highest population
        for n in neighbors:
            # normalize number of refugees at each node
            refpop_norm = self.graph.nodes[n]['weight'] / self.max_pop
            refugees_at_node = self.graph.nodes[n]['weight']
            kin_at_node = kin_nodes.count(n)
            friends_at_node = friend_nodes.count(n)
            location_score = self.graph.nodes[n]['location_score']
            # neighbord desirability calculated for each neighbor
            desirability = refpop_norm + kin_at_node + (friends_at_node / 2) + location_score
            # print(n, count_kin_at_node, count_friends_at_node, location_score, desirability)
            if desirability > most_desirable_score:
                most_desirable_score = desirability
                most_desirable_neighbor = n

        return most_desirable_neighbor

    def process_refs(self, refs):
        print('Starting process...')

        # define new refugee list per chunk of refs
        new_refs = []
        # dictionary for how many refugees are moving in and out of each node; initialized at 0
        new_weights = {key: 0 for key in self.graph.nodes}

        for x, ref in enumerate(refs):
            node = self.all_refugees[ref].node
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

            new_refs.append(copy.deepcopy(self.all_refugees[ref]))

            if not move:
                continue

            # Create a copy of refugee, assign new node attribute
            new_node = self.find_new_node(node, ref)
            if new_node is None:
                continue
            # update refugee with new node location
            new_refs[x].node = new_node

            # Update weight dict for every node, subtracting the refugee that left and adding the refugee that entered
            new_weights[node] -= 1
            new_weights[new_node] += 1

        # return new refugee list and node weight updates for these refs
        return new_refs, new_weights

    def step(self, step):
        orig_weights = nx.get_node_attributes(graph, 'weight')

        self.max_pop = max(orig_weights.values())
        start = time.time()
        if NUM_CHUNKS > 1:
            print('Multiprocessing...')
            chunked_refs = np.array_split([x for x in range(len(self.all_refugees))], NUM_CHUNKS)
            pool = Pool(NUM_CHUNKS)
            results = pool.map(self.process_refs, chunked_refs)
            pool.close()
            pool.join()
        else:
            print('Not Multiprocessing')
            results = [self.process_refs([x for x in range(len(self.all_refugees))])]

        print('time to process -', time.time() - start)
        # self.all_refugees = []
        # new_weights = []
        # new_weights.append(orig_weights)
        # for result in results:
        #     self.all_refugees.extend(result[0])
        #     new_weights.append(result[1])
        #
        # # self.all_refugees = new_refs
        #
        # new_weights = pd.DataFrame(new_weights)
        #
        # new_weights = dict(zip(self.graph.nodes, list(new_weights.sum(numeric_only=True))))
        #
        # nx.set_node_attributes(self.graph, new_weights, 'weight')
        #
        # # seed border crossing nodes with new refugees
        # new_ref_index = self.num_refugees
        # self.num_refugees += SEED_REFS * len(BORDER_CROSSING_LIST)
        # for node in BORDER_CROSSING_LIST:
        #     self.graph.nodes[node]['weight'] += SEED_REFS
        #     self.all_refugees.extend([Ref(node, self.num_refugees) for x in range(0, SEED_REFS)])
        #
        # for index in range(new_ref_index, self.num_refugees):
        #     self.all_refugees[index].create_social_links(index, self)

    def run(self):
        # Add status bar for simulation run
        avg_step_time = 0
        for x in list(range(self.num_steps)):
            start = time.time()
            print(f'Starting step {x+1}')
            self.step(x)
            step_time = time.time() - start
            avg_step_time += step_time
            print(f'Step Time: {step_time:2f}')
        print(f'Average step time: {step_time:2f}')


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

    graph, polys = build_graph()

    # graph = nx.complete_graph(40)
    #
    # nx.set_node_attributes(graph, name='weight', values=5000)
    # nx.set_node_attributes(graph, name='num_conflicts', values=0)
    # nx.set_node_attributes(graph, name='num_camps', values=1)
    # nx.set_node_attributes(graph, name='location_score', values=0.5)

    # Run Sim
    # Set number of simulation steps; 1 step = 1 day
    num_steps = 1
    sim = Sim(graph, num_steps)

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
