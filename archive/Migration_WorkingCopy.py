"""
ABMs in the Study of Forced Migration
Author: Melonie Richey
Last Edited: 2020

Spatially-Explicit Agent-Based Model (ABM) of Forced Migration from Syria to Turkey ###
"""


import random
import math
import networkx as nx
import geopandas as gpd
import unidecode
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing as mp

# CONSTANTS
# Percentage of refugees that move if in a district with one or more refugee camps
PERCENT_MOVE_AT_CAMP = 0.3
# Percentage of refugees that move if in a district with one of more conflict events
PERCENT_MOVE_AT_CONFLICT = 1
# Percentage of refugees that move if in a district without a conflict event or a camp
PERCENT_MOVE_AT_OTHER = 0.7
# Number of refugees that cross the Syrian-Turkish border at each time step
SEED_REFS = 10
# Districts that contain open border crossings during month of simulation start
BORDER_CROSSING_LIST = ['Merkez Kilis', 'KarkamA+-A', 'YayladaAA+-', 'Kumlu']
# # Point to calculate western movement
LONDON_COORDS = (51.5074, -0.1278)


class Ref(object):
    """
    Class representative of a single refugee
    """
    def __init__(self, agent_id, sim):
        self.agentID = agent_id  # Not used. Agent ID == Index in Sim.all_refugees
        self.kin_list = {}
        self.friend_list = {}

        # create kin
        num_kin = 1  # random.randint(1,3)
        for x in range(num_kin):
            kin = x
            while kin != x:
                kin = random.randint(0, sim.num_refugees)
            self.kin_list[kin] = 1
            # set for other kin
            # self.all_refugees[kin].kin_list[r] = 1

        # create friends
        num_friend = 1  # random.randint(1,3)
        for x in range(num_friend):
            friend = x
            while friend != x:
                friend = random.randint(0, sim.num_refugees)
            self.friend_list[friend] = 1
            # set for other kin
            # self.all_refugees[friend].friend_list[r] = 1


class Sim(object):
    """
    Class representative of the simulation
    """
    def __init__(self, num_steps, graph):
        self.num_steps = num_steps
        self.graph = graph
        node_weights = [self.graph.nodes[n]['weight'] for n in self.graph.nodes]
        self.num_refugees = sum(node_weights)
        self.all_refugees = [Ref(x, self) for x in range(0, self.num_refugees)]
        self.max_pop = None

        for n, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['refs'] = list(range(sum(node_weights[:n]), sum(node_weights[:n + 1])))

    def sim_step(self, step):

        # Calculate normalized refpop at each iteration
        weights = [self.graph.nodes[n]['weight'] for n in self.graph.nodes.keys()]
        self.max_pop = max(weights)
        for node in self.graph.nodes:
            self.step(node, step)

            if node in BORDER_CROSSING_LIST:
                # seed network
                # print(node, self.graph.nodes[node]['weight'])
                start_index = self.num_refugees
                self.num_refugees = len(self.all_refugees) + SEED_REFS
                new_refugees = [Ref(x, self) for x in range(0, SEED_REFS)]
                self.all_refugees += new_refugees
                self.graph.nodes[node]['incoming_refs'] += range(start_index, start_index + SEED_REFS)

        print('Updating weights...')
        # Update weights in network
        for node in tqdm(self.graph.nodes):
            # Update ref list for each node
            self.graph.nodes[node]['refs'] = list(
                filter(lambda x: x not in self.graph.nodes[node]['outgoing_refs'], self.graph.nodes[node]['refs']))
            self.graph.nodes[node]['refs'] += self.graph.nodes[node]['incoming_refs']

            # Reset incoming and outgoing refs
            self.graph.nodes[node]['incoming_refs'] = []
            self.graph.nodes[node]['outgoing_refs'] = []

            # Update network weight
            self.graph.nodes[node]['weight'] = len(self.graph.nodes[node]['refs'])

        #     # Update social network weights
        #     # TODO - add friend_list updates
        #     refugees = self.graph.nodes[node]['refs']
        #     print('\nNode weights updated for {}...'.format(node))
        #     print('Updating friendship network for {}...'.format(node))
        #     # print(len(refugees))
        #     for r1, ref1 in enumerate(tqdm(refugees)):
        #         # Add 1 if kin exists or delete if not
        #         keys = self.all_refugees[ref1].kin_list.keys()
        #         # print(len(keys))
        #         keys_to_delete = []
        #         for key in keys:
        #             if key in refugees:
        #                 self.all_refugees[ref1].kin_list[key] += 1
        #             else:
        #                 keys_to_delete.append(key)
        #
        #         for key in keys_to_delete:
        #             del self.all_refugees[ref1].kin_list[key]
        #
        #         # Add new kin
        #         for r2, ref2 in enumerate(refugees):
        #             if ref2 not in self.all_refugees[ref1].kin_list:
        #                 self.all_refugees[ref1].kin_list[ref2] = 1
        #
        # print('Kin updated')

    def step(self, node, step):
        # print('In step. Node - {}'.format(node))

        # Set variables for count of conflict events and camps in each district
        num_conflicts = self.graph.nodes[node]['num_conflicts']
        num_camps = self.graph.nodes[node]['num_camps']

        if num_conflicts > 0:
            # Conflict zone - 1.0 move chance
            for ref in self.graph.nodes[node]['refs']:
                self.move_refugee(node, ref)

        elif num_camps:
            # At a camp - 0.003 move chance
            for ref in self.graph.nodes[node]['refs']:
                # Check if we should move
                move = random.random() < PERCENT_MOVE_AT_CAMP
                if move:
                    self.move_refugee(node, ref)
        else:
            # Neither camp nor conflict - 0.5 move chance
            for ref in self.graph.nodes[node]['refs']:
                # Check if we should move
                move = random.random() < PERCENT_MOVE_AT_OTHER
                if move:
                    self.move_refugee(node, ref)

    def move_refugee(self, node, ref):
        # move to a node
        # Find neighbor with highest weight
        neighbors = list(self.graph.neighbors(node))

        # Initialize max node value to negative number
        most_desirable_score = -99
        most_desirable_neighbor = None

        # check to see if there are neighbors (in case node is isolate)
        if len(neighbors) == 0:
            # print(ref_pop, "refugees can't move from isolates", node)
            return

        # calculate neighbor with highest population
        for n in neighbors:
            refpop_norm = self.graph.nodes[n]['weight'] / self.max_pop
            refugees_at_node = self.graph.nodes[n]['refs']
            count_kin_at_node = len([x for x in self.all_refugees[ref].kin_list if x in refugees_at_node])
            count_friends_at_node = len([x for x in self.all_refugees[ref].friend_list if x in refugees_at_node])
            location_score = self.graph.nodes[n]['location_score']
            desirability = refpop_norm + count_kin_at_node + (count_friends_at_node / 2) + location_score
            # print(n, count_kin_at_node, count_friends_at_node, location_score, desirability)
            if desirability > most_desirable_score:
                most_desirable_score = desirability
                most_desirable_neighbor = n

        # print(most_desirable_neighbor, most_desirable_score)
        self.graph.nodes[most_desirable_neighbor]['incoming_refs'].append(ref)
        self.graph.nodes[node]['outgoing_refs'].append(ref)

    def run(self):
        # Add status bar for simulation run
        for x in list(range(self.num_steps)):
            start = time.time()
            print('Starting step {}'.format(x))
            self.sim_step(x)
            print('Step Time: {:2f}'.format(time.time() - start))


def build_graph():
    # ***Data Engineering Using GeoPandas***

    # Get shapefile of Turkish districts (and re-project)
    polys = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\Input\gadm36_TUR_2.shp")

    # Check and set spatial projection
    # polys = polys.to_crs({'init': 'epsg:4326'})
    # print(polys.crs)

    # Get refugee population by province data (and re-project), from Turkish Statistical Institute, February 2019
    pop_by_province = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\Input\REFPOP.shp")

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
    polys.to_file("output_1.shp")

    # ***Create Additional Spatial Data Layers***

    # Create Conflict data layer
    # Read in ACLED event data from February 2019 (and set projection)
    df_conflict = pd.read_csv(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\Input\FebACLEDextract.csv")
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
    camps = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\Input\tur_camps.shp")
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
    points.to_file('output_2_centroids.shp')

    # ***Add Edges to Graph***
    for edge in edges:
        graph.add_edge(edge[0], edge[1], weight=1)

    # ***Draw Graph***
    base = polys.plot(color='cadetblue', edgecolor='black')
    nx.draw(graph, node_size=25, node_color='darkblue', pos=positions)
    # plt.show()

    return graph, polys


if __name__ == '__main__':
    """
    Program Execution starts here
    """

    graph, polys = build_graph()

    # Run Sim
    # Set number of simulation steps; 1 step = 1 day
    num_steps = 7
    sim = Sim(num_steps, graph)

    start_node_weights = [graph.nodes[n]['weight'] for n in graph.nodes]
    sim.run()
    end_node_weights = [graph.nodes[n]['weight'] for n in graph.nodes]

    # print starting and ending node weights
    for node, s_weight, e_weight in zip(graph.nodes, start_node_weights, end_node_weights):
        print(node, s_weight, e_weight)

    print("Total start weight:", sum(start_node_weights))
    print("Total end weight:", sum(end_node_weights))

    # Write end_node_weights to final_REFPOP column in shapefile
    output = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\output_1.shp")

    for index, row in output.iterrows():
        # Convert ending node weights to integer
         for i in range(len(end_node_weights)):
            end_node_weights[i] = int(end_node_weights[i])
            output.at[index, "simEnd"] = int(end_node_weights[index])

    # write out to shapefile
    output.to_file("output_3_simOutput.shp")

    # visualize simulation output
    #output = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\Summer 2019 Comp Migration\output_3_simOutput.shp")
    #colors = 6
    #figsize = (26, 20)
    #cmap = 'winter_r'
    #simEnd = output.simEnd
    #output.plot(column=simEnd, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)

    ## MODEL VALIDATION ##
    val = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\Input\gadm36_TUR_1_val.shp")
    polys = gpd.read_file(r"C:\Users\mrich\OneDrive\GMU\2020 Dissertation\Input\REFPOP.shp")
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
    val.to_file("output_4.shp")

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
    output.to_file("output_5_validation.shp")

    # visualize validation output
    validation = gpd.read_file(r"C:\Users\mrich\GMU\2020 Dissertation\output_5_validation.shp")
    colors = 6
    figsize = (26, 20)
    cmap = 'winter_r'
    accuracy = validation.accuracy
    validation.plot(column=accuracy, cmap=cmap, scheme='equal_interval', k=colors, legend=True, linewidth=10)

    # SHOW PLOTS
    plt.show()