"""
ABMs in the Study of Forced Migration
Author: Melonie Richey
Last Edited: 2020

Spatially-Explicit Agent-Based Model (ABM) of Forced Migration from Syria to Turkey ###
"""
import os
import sys
import csv
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
# For testing
TEST = False  # True
NUM_NODES = 4
AVG__PHYSICAL_CONNECTIONS = 3
TOTAL_NUM_REFUGEES = 4  # refs per node = TOTAL_NUM_REFUGEES / NUM_NODES

# For timing different num processes
TIME_TRIAL = False

# Whether to pre-process the data
PREPROCESS = True

# Location of data
DATA_DIR = './data'

# Whether to visualize the geographic network
DRAW_GEO = False
# Whether to print the node weights at the end of running
PRINT_NODE_WEIGHTS = False
# Set number of simulation steps; 1 step = 1 day
NUM_STEPS = 1

# Number of friendships and kin to create
NUM_FRIENDS = random.randint(1, 3)
NUM_KIN = random.randint(1, 3)

# Percentage of refugees that move if in a district with one or more refugee camps
PERCENT_MOVE_AT_CAMP = 0.3
# Percentage of refugees that move if in a district with one of more conflict events
PERCENT_MOVE_AT_CONFLICT = 1
# Percentage of refugees that move if in a district without a conflict event or a camp
PERCENT_MOVE_AT_OTHER = 0.7

# Point to calculate western movement
LOCATION = (51.5074, -0.1278)

# Weight of each of the node desirability variables
POPULATION_WEIGHT = .25  # total number of refs at node
LOCATION_WEIGHT = .25  # closeness to LOCATION point
CAMP_WEIGHT = .25  # (camps * CAMP_WEIGHT)
CONFLICT_WEIGHT = .25  # (conflicts * (-1) * CONFLICT_WEIGHT)
KIN_WEIGHT = .25  # (num kin * FRIEND_WEIGHT)
FRIEND_WEIGHT = .25  # (num friends * KIN_WEIGHT)

# Number of refugees that cross the Syrian-Turkish border at each time step
SEED_REFS = 10
# Districts that contain open border crossings during month of simulation start
BORDER_CROSSING_LIST = ['Merkez Kilis', 'KarkamA+-A', 'YayladaAA+-', 'Kumlu']  # [0, 1, 2, 3]

# How many friend relationships to add per node
NEW_FRIENDS_LOWER = 0
NEW_FRIENDS_UPPER = 5

# Number of chunks (processes) to split refugees into during a sim step
# These dont necessarily have to be equal
NUM_CHUNKS = 4
NUM_PROCESSES = 4  # mp.cpu_count()


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
            while kin == index:
                kin = random.randint(0, sim.num_refugees - 1)
            self.kin_list[kin] = 1
            # set for other kin
            sim.all_refugees[kin].kin_list[index] = 1

        # create friends
        for x in range(NUM_FRIENDS):
            friend = index
            while friend == index:
                friend = random.randint(0, sim.num_refugees - 1)
            self.friend_list[friend] = 1
            # set for other friend
            sim.all_refugees[friend].friend_list[index] = 1


class Sim(object):
    """
    Class representative of the simulation
    """

    def __init__(self, graph, num_steps=10, num_processes=1, num_chunks=1):
        self.graph = graph
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.num_chunks = num_chunks
        self.num_refugees = sum([self.graph.nodes[n]['weight'] for n in self.graph.nodes])
        self.all_refugees = []
        for node in self.graph.nodes():
            self.all_refugees.extend([Ref(node, self.num_refugees) for x in range(self.graph.nodes[node]['weight'])])
        for index, ref in enumerate(self.all_refugees):
            ref.create_social_links(index, self)

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

        kin_nodes = [self.all_refugees[kin].node for kin in self.all_refugees[ref].kin_list]
        friend_nodes = [self.all_refugees[friend].node for friend in self.all_refugees[ref].friend_list]

        # calculate neighbor with highest population
        for n in neighbors:
            kin_at_node = kin_nodes.count(n)
            friends_at_node = friend_nodes.count(n)
            desirability = (kin_at_node * KIN_WEIGHT) + (friends_at_node * FRIEND_WEIGHT) + self.graph.nodes[n]['node_score']  # + self.graph.nodes[n]['']
            if desirability > most_desirable_score:
                most_desirable_score = desirability
                most_desirable_neighbor = n

        return most_desirable_neighbor

    def process_refs(self, refs):
        new_refs = []
        ref_nodes = {key: [] for key in self.graph.nodes}

        for x, ref in enumerate(refs):
            node = self.all_refugees[ref].node
            num_conflicts = self.graph.nodes[node]['num_conflicts']
            num_camps = self.graph.nodes[node]['num_camps']

            if num_conflicts > 0:
                # Conflict zone
                move = True
            elif num_camps > 0:
                # At a camp
                move = random.random() < PERCENT_MOVE_AT_CAMP
            else:
                # Neither camp nor conflict
                move = random.random() < PERCENT_MOVE_AT_OTHER

            new_refs.append(copy.deepcopy(self.all_refugees[ref]))

            if move:
                new_node = self.find_new_node(node, ref)
                if new_node:
                    new_refs[x].node = new_node

            # Add ref to its new node
            ref_nodes[new_refs[x].node].append(ref)

        # return new refugee list and node weight updates for these refs
        return new_refs, ref_nodes

    def step(self):
        nx.get_node_attributes(self.graph, 'weight')
        orig_weights = nx.get_node_attributes(self.graph, 'weight')

        # Update normalized node weights
        max_pop = max(orig_weights.values())
        norm_weights = [x / max_pop for x in orig_weights.values()]
        norm_weights = dict(zip(orig_weights.keys(), norm_weights))
        nx.set_node_attributes(self.graph, norm_weights, 'norm_weight')

        # Normalize camps
        num_camps = nx.get_node_attributes(self.graph, 'num_camps')
        max_camps = max(list(num_camps.values()) + [1])
        num_camps = [x / max_camps for x in num_camps.values()]

        # Normalize conflicts
        num_conflicts = nx.get_node_attributes(self.graph, 'num_conflicts')
        max_conflict = max(list(num_conflicts.values()) + [1])  # Add a max of 1 to prevent division by zero error
        num_conflicts = [x / max_conflict for x in num_conflicts.values()]

        # Update node score
        location_scores = nx.get_node_attributes(self.graph, 'location_score')
        node_scores = [
            (w * POPULATION_WEIGHT) + (x * LOCATION_WEIGHT) + (y * CAMP_WEIGHT) + (z * (-1) * CONFLICT_WEIGHT) for
            w, x, y, z in
            zip(norm_weights.values(), location_scores.values(), num_camps, num_conflicts)]
        node_scores = dict(zip(norm_weights.keys(), node_scores))
        nx.set_node_attributes(self.graph, node_scores, 'node_score')

        # Whether to process in parallel or synchronously
        if self.num_processes > 1:
            print(f'Staring {self.num_processes} processes...')
            # Chunk refs and send to processes
            chunked_refs = np.array_split([x for x in range(len(self.all_refugees))], self.num_chunks)

            pool = Pool(self.num_processes)

            results = pool.map(self.process_refs, chunked_refs)
            pool.close()
            pool.join()
        else:
            print('Not Multiprocessing')
            results = [self.process_refs([x for x in range(len(self.all_refugees))])]

        print('Gathering results...')
        self.all_refugees = []
        ref_nodes = []
        for result in results:
            self.all_refugees.extend(result[0])
            ref_nodes.append(result[1])

        print('Updating node refugee lists and counts...')
        # Update node refugee lists (contains the indices of refugees at node)
        ref_nodes = pd.DataFrame(ref_nodes)
        ref_nodes = dict(zip(self.graph.nodes, list(ref_nodes.sum())))
        new_weights = [len(x) for x in ref_nodes.values()]
        # Update node weights
        new_weights = dict(zip(self.graph.nodes, new_weights))
        nx.set_node_attributes(self.graph, new_weights, 'weight')

        print("Adding friendships at camps...")
        # Randomly create friendships between refs at same node
        new_friendships = 0
        for node in self.graph.nodes():
            if (self.graph.nodes[node]['num_camps'] > 0) and (self.graph.nodes[node]['weight'] > 1):
                num_new_rels = random.randint(NEW_FRIENDS_LOWER, NEW_FRIENDS_UPPER)
                for x in range(num_new_rels):
                    ref1 = random.choice(ref_nodes[node])
                    ref2 = ref1
                    while ref2 == ref1:
                        ref2 = random.choice(ref_nodes[node])
                    new_friendships += 1
                    self.all_refugees[ref1].friend_list[ref2] = 1
                    self.all_refugees[ref2].friend_list[ref1] = 1
        print(f'Added {new_friendships} friendships at camps...')

        print('Seeding network at border crossings...')
        new_ref_index = self.num_refugees
        self.num_refugees += SEED_REFS * len(BORDER_CROSSING_LIST)
        for node in BORDER_CROSSING_LIST:
            self.graph.nodes[node]['weight'] += SEED_REFS
            self.all_refugees.extend([Ref(node, self.num_refugees) for x in range(0, SEED_REFS)])

        for index in range(new_ref_index, self.num_refugees):
            self.all_refugees[index].create_social_links(index, self)

    def run(self):
        # Add status bar for simulation run
        avg_step_time = 0
        for x in list(range(self.num_steps)):
            start = time.time()
            print(f'Starting step {x + 1}...')
            self.step()
            step_time = time.time() - start
            avg_step_time += step_time
            print(f'Step Time: {step_time:2f}')

        avg_step_time /= self.num_steps
        print(f'Average step time: {avg_step_time:2f}')

        return avg_step_time


def preprocess():
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

    # ** Remove non-english characters **
    # ** fix duplicate district names ("Merkez") **
    new_names = []  # for tracking nodes with same name. We will append index in dataframe if not unique
    for index, row in polys.iterrows():
        name = unidecode.unidecode(row.NAME_2)
        if name in "Merkez":
            name += " " + unidecode.unidecode(row.NAME_1)
        if name in new_names:
            name += str(index)
        new_names.append(name)
    polys["NAME_2"] = new_names

    # ** Add Population Data **
    polys["REFPOP"] = 0
    # Assign an equal portion of refs to each node in province
    for index, row in pop_by_province.iterrows():
        # print(row['REFPOP'], row['count'])
        refs_at_node = row['REFPOP'] / row['count']
        if math.isnan(refs_at_node):
            refs_at_node = 0
        polys.loc[polys.NAME_1 == row.NAME_1, 'REFPOP'] = int(refs_at_node)

    # ** Add conflict data **
    # Read in ACLED event data from February 2019 (and set projection)
    df_conflict = pd.read_csv(os.path.join(DATA_DIR, 'FebACLEDextract.csv'))
    conflict = gpd.GeoDataFrame(df_conflict,
                                geometry=gpd.points_from_xy(df_conflict.longitude, df_conflict.latitude),
                                crs='epsg:4326')
    # conflict.crs = 'epsg:4326'
    # Create new column in target shapefile for count of conflict events per district
    polys["conflict"] = polys.apply(lambda row: sum(conflict.within(row.geometry)), axis=1)

    # ** Add camps **
    # Read in refugee camp data (and re-project) from UNHCR Regional IM Working Group February 2019 (updated every 6 months)
    camps = gpd.read_file(os.path.join(DATA_DIR, 'tur_camps.shp'))
    # Create new column in target shapefile for refugee camps
    polys["camp"] = polys.apply(lambda row: sum(camps.within(row.geometry)), axis=1)

    ## Add location score
    # Calculate location score. Districts closest to specified location are scored highest.
    polys['location'] = polys.apply(
        lambda row: math.sqrt(
            (row.geometry.centroid.x - LOCATION[1]) ** 2 + (row.geometry.centroid.y - LOCATION[0]) ** 2), axis=1)
    max_distance = max(list(polys['location']))
    polys['location'] = polys.apply(lambda row: 1 - (row.location / max_distance), axis=1)

    ## Create centroids GPD
    points = polys.copy()
    points['geometry'] = points['geometry'].centroid

    # Write points to new Shapefile
    points.to_file(os.path.join(DATA_DIR, 'preprocessed_data.shp'))

    # Write polys to new Shapefile
    polys.to_file(os.path.join(DATA_DIR, 'preprocessed_poly_data.shp'))

    return polys, points


def build_graph(data):
    # ***Network Creation using NetworkX***
    graph = nx.Graph()

    if isinstance(data, str):
        data = gpd.read_file(data)

    # ***Add Nodes to Graph***
    positions = {}
    for index, row in data.iterrows():
        node = row.NAME_2
        # Add the coordinates to the nodes so they can be displayed geospatially
        coords = row['geometry'].centroid
        graph.add_node(node, pos=coords, weight=row.REFPOP,
                       num_conflicts=row.conflict, num_camps=row.camp,
                       location_score=row.location)

        positions[node] = (coords.x, coords.y)

        nx.set_node_attributes(graph, positions, 'position')

        # ***Add Edges to Graph***
        neighbors = data[data.geometry.touches(row['geometry'])].NAME_2.tolist()
        for n in neighbors:
            edge = (row.NAME_2, n)
            edge = sorted(edge)
            graph.add_edge(edge[0], edge[1], weight=1)

    return graph


def draw(polys, graph):
    polys.plot(color='cadetblue', edgecolor='black')
    nx.draw(graph, node_size=25, node_color='darkblue', pos=nx.get_node_attributes(graph, 'position'))
    plt.show()


def time_trial(graph, output_file='results.csv', num_steps=10, num_processes=[1], num_batches=[1]):
    with open(output_file, 'w+', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['STEPS', 'PROCESSES', 'BATCHES', 'TIME_(S)'])
        for n_process in num_processes:
#             for n_batch in num_batches:
            sim = Sim(graph, num_steps, n_process, n_process)
            avg_step_time = sim.run()

            writer.writerow([num_steps, n_process, n_process, avg_step_time])
            fp.flush()


if __name__ == '__main__':
    """
    Program Execution starts here
    """

    if TEST:
        print('Building test graph...')
        # For testing
        graph = nx.fast_gnp_random_graph(NUM_NODES, float(AVG__PHYSICAL_CONNECTIONS) / NUM_NODES)

        nx.set_node_attributes(graph, name='weight', values=int(TOTAL_NUM_REFUGEES / NUM_NODES))
        nx.set_node_attributes(graph, name='num_conflicts', values=0)  # todo - can make this random
        nx.set_node_attributes(graph, name='num_camps', values=0)  # todo - can make this random
        nx.set_node_attributes(graph, name='location_score', values=0.5)  # todo - can make this random
    else:
        if PREPROCESS:  # Run pre-processing
            # Option 1 - Preprocess shapefiles
            print('Pre-processing graph data...')
            start = time.time()
            polys, points, = preprocess()
            print(f'Completed in {time.time() - start:.2f}s...')
        else:
            # Option 2 - Build graph from preprocessed polys shapefile
            print('Loading graph data from file...')
            start = time.time()
            polys = gpd.read_file('./data/preprocessed_poly_data.shp')
            print(f'Completed in {time.time() - start:.2f}s...')

        graph = build_graph(polys)

        # Draw graph
        if DRAW_GEO:
            print('Drawing graph...')
            draw(polys, graph)

    if TIME_TRIAL:

        num_steps = 5
        processes = [1, 2, 4, 8, 16]  # , 4, 8, 12, 16]
        chunks = [1]  # , 4, 8, 12, 16]

        time_trial(graph, num_steps=num_steps, num_processes=processes, num_batches=chunks)
        sys.exit()

    # Run Sim
    print('Creating sim...')
    start = time.time()
    sim = Sim(graph, NUM_STEPS, NUM_PROCESSES, NUM_CHUNKS)
    print(f'Created sim in {time.time() - start:.2f}s...')

    start_node_weights = nx.get_node_attributes(graph, 'weight')
    sim.run()
    end_node_weights = start_node_weights #  nx.get_node_attributes(graph, 'weight')
    if PRINT_NODE_WEIGHTS:
        for node in graph.nodes:
            print(node, start_node_weights[node], end_node_weights[node])

    print("Total start weight:", sum(start_node_weights.values()))
    print("Total end weight:", sum(end_node_weights.values()))

    # Write out to shapefile
    polys['simEnd'] = polys['NAME_2'].map(end_node_weights)
    polys.to_file(os.path.join(DATA_DIR, 'simOutput.shp'))

    # visualize simulation output
    # output = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\Summer 2019 Comp Migration\output_3_simOutput.shp")
    # colors = 6
    # figsize = (26, 20)
    # cmap = 'winter_r'
    # simEnd = output.simEnd
    # output.plot(column=simEnd, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)

    ## MODEL VALIDATION ##
    val = gpd.read_file(os.path.join(DATA_DIR, 'gadm36_TUR_1_val.shp'))
    pop_by_province = gpd.read_file(os.path.join(DATA_DIR, 'REFPOP.shp'))
    # sim_val = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\Summer 2019 Comp Migration\output_3_simOutput.shp")

    # Remove non-english characters and fix duplicate district names ("Merkez")
    for index, row in val.iterrows():
        name = unidecode.unidecode(row.NAME_1)
        val.at[index, "NAME_1"] = name


    # Take refugee population by province, divide by number of districts in province, assign each equivalent value as REFPOP of district
    polys['valPop'] = 0
    for index, row in val.iterrows():
        # print(type(row['val_mar19']))
        # print(type(polys_val.loc[index].count))

        val_calc = row['val_mar19'] / float(pop_by_province.loc[index]['count'])
        # print(polys.NAME_1)
        if not math.isnan(val_calc):
            val_calc = int(val_calc)

            polys.valPop.iloc[[polys.NAME_1 == row.NAME_1]] = val_calc

    # Normalize both actual and predicted REFPOP for district-level comparison
    minPop = min(polys.valPop)
    maxPop = max(polys.valPop)
    polys['valPopNorm'] = (polys['valPop'] - minPop) / (maxPop - minPop)
    minPop = min(polys.simEnd)
    maxPop = max(polys.simEnd)
    polys['simEnd_norm'] = (polys['simEnd'] - minPop) / (maxPop - minPop)
    # print(polys.simEnd_norm)

    # Comparative scaled_actual & scale_predicted
    polys['accuracy'] = (polys.simEnd_norm - polys.valPopNorm)

    # Write out new shapefile with validation accuracy by district
    polys.to_file(os.path.join(DATA_DIR, 'validationResults.shp'))

    # visualize validation output - unnecessary to read again
    # validation = gpd.read_file(os.path.join(DATA_DIR, 'output_5_validation.shp'))
    colors = 6
    figsize = (26, 20)
    cmap = 'winter_r'
    accuracy = polys.accuracy
    polys.plot(column=accuracy, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)

    # SHOW PLOTS
    plt.show()
