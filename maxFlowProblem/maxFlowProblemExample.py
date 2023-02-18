#maxFlowProblemExample

import ortools
import numpy as np


from ortools.graph.python import max_flow

"""MaxFlow simple interface example."""
# Instantiate a SimpleMaxFlow solver.
smf = max_flow.SimpleMaxFlow()

# Define three parallel arrays: start_nodes, end_nodes, and the capacities
# between each pair. For instance, the arc from node 0 to node 1 has a
# capacity of 20.
# start_nodes = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
# end_nodes = np.array([1, 2, 3, 2, 4, 3, 4, 2, 4])
# capacities = np.array([20, 30, 10, 40, 30, 10, 20, 5, 20])

start_nodes = np.array([0, 0, 1,  1, 1,  2,  2, 3,  4])
end_nodes = np.array([1,   2, 2,  3, 4,  3,  4, 4,  3])
capacities = np.array([15, 8, 20, 4, 10, 15, 4, 20, 5])

# Add arcs in bulk.
#   note: we could have used add_arc_with_capacity(start, end, capacity)
all_arcs = smf.add_arcs_with_capacity(start_nodes, end_nodes, capacities)

# Find the maximum flow between node 0 and node 4.
status = smf.solve(0, 4)

if status != smf.OPTIMAL:
    print('There was an issue with the max flow input.')
    print(f'Status: {status}')
    exit(1)
print('Max flow:', smf.optimal_flow())
print('')
print(' Arc    Flow / Capacity')
solution_flows = list()
for i in np.arange(len(all_arcs)):
    solution_flows.append(smf.flow(all_arcs[i]))
for arc, flow, capacity in zip(all_arcs, solution_flows, capacities):
    print(f'{smf.tail(arc)} / {smf.head(arc)}   {flow:3}  / {capacity:3}')
print('Source side min-cut:', smf.get_source_side_min_cut())
print('Sink side min-cut:', smf.get_sink_side_min_cut())





#### Plotting the directed graph
import igraph as ig
import matplotlib.pyplot as plt


# Create a directed graph
g = ig.Graph(directed=True)# Add 5 vertices
num = 18
g.add_vertices(num)
# Add ids and labels to vertices
for i in range(len(g.vs)):
    g.vs[i]["id"]= i
    g.vs[i]["label"]= str(i)# Add edges
# g.add_edges([(0,2),(0,1),(0,3),(1,2),(1,3),(2,4),(3,4)])# Add weights and edge labels
# weights = [8,6,3,5,6,4,9]
numLinks = int(np.floor(18*17*0.3)) # let's say only 50% of links are available
starts = np.random.randint(0,num,numLinks)
stops = np.random.randint(1,num,numLinks)
edgeList = list()
for i in np.arange(numLinks):
    if starts[i] == stops[i]: #skip any that are the same start and stop
        continue
    edgeList.append((starts[i],stops[i]))
edgeListSet = set(edgeList) #remove duplicates
edgeList = list(edgeList)
g.add_edges(edgeList)# Add weights and edge labels
weights = np.random.randint(1,20,len(edgeList)).tolist()
g.es['weight'] = weights
g.es['label'] = weights


visual_style = {}
out_name = "graph.png"
# Set bbox and margin
visual_style["bbox"] = (800,800)
visual_style["margin"] = 1
# Set vertex colours
visual_style["vertex_color"] = 'white'
# Set vertex size
visual_style["vertex_size"] = 0.2
# Set vertex lable size
visual_style["vertex_label_size"] = 10
# Don't curve the edges
visual_style["edge_curved"] = False
visual_style["edge_width"] = 1#[1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
visual_style["rescale"] = False
# Set the layout
#my_layout = g.layout_lgl()
#my_layout = g.layout_umap(dist=None, weights=None, dim=2, seed=None, min_dist=0.2, epochs=500)
#my_layout = g.layout_davidson_harel(seed=None, maxiter=40, fineiter=10, cool_fact=0.5, weight_node_dist=10, weight_border=0.0,\
#    weight_edge_lengths=0.1, weight_edge_crossings=10, weight_node_edge_dist=-1)
my_layout = g.layout_drl(weights=1/np.asarray(weights), fixed=None, seed=None, options={"dge_cut":0.7}, dim=2)
visual_style["layout"] = my_layout# Plot the graph
#visual_style["layout"] = "circle"
#visual_style["layout"] = "random"
#visual_style["layout"] = "star"
fig, ax = plt.subplots()
visual_style["target"] = ax
#ig.plot(g, out_name, **visual_style)
ig.plot(g, **visual_style)
plt.show(block=False)




print("Number of vertices in the graph:", g.vcount())
print("Number of edges in the graph", g.ecount())
print("Is the graph directed:", g.is_directed())
print("Maximum degree in the graph:", g.maxdegree())
print("Adjacency matrix:\n", g.get_adjacency())







###############################################################################
# Create a directed graph
g2 = ig.Graph(directed=True)# Add 5 vertices
num = 18
g2.add_vertices(num)
# Add ids and labels to vertices
for i in range(len(g2.vs)):
    g2.vs[i]["id"]= i
    g2.vs[i]["label"]= str(i)# Add edges
# g.add_edges([(0,2),(0,1),(0,3),(1,2),(1,3),(2,4),(3,4)])# Add weights and edge labels
# weights = [8,6,3,5,6,4,9]
numLinks = int(np.floor(18*17*0.3)) # let's say only 50% of links are available
starts = np.random.randint(0,num,numLinks)
stops = np.random.randint(1,num,numLinks)
edgeList = list()
for i in np.arange(numLinks):
    if starts[i] == stops[i]: #skip any that are the same start and stop
        continue
    edgeList.append((starts[i],stops[i]))
edgeListSet = set(edgeList) #remove duplicates
edgeList = list(edgeList)
g2.add_edges(edgeList)# Add weights and edge labels
weights = np.random.randint(1,20,len(edgeList)).tolist()
g2.es['weight'] = weights
g2.es['label'] = weights


visual_style = {}
out_name = "graph2.png"
# Set bbox and margin
visual_style["bbox"] = (800,800)
visual_style["margin"] = 1
# Set vertex colours
visual_style["vertex_color"] = 'white'
# Set vertex size
visual_style["vertex_size"] = 0.2
# Set vertex lable size
visual_style["vertex_label_size"] = 10
# Don't curve the edges
visual_style["edge_curved"] = False
visual_style["edge_width"] = 1#[1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
visual_style["rescale"] = False
# Set the layout
#my_layout = g.layout_lgl()
#my_layout = g.layout_umap(dist=None, weights=None, dim=2, seed=None, min_dist=0.2, epochs=500)
#my_layout = g.layout_davidson_harel(seed=None, maxiter=40, fineiter=10, cool_fact=0.5, weight_node_dist=10, weight_border=0.0,\
#    weight_edge_lengths=0.1, weight_edge_crossings=10, weight_node_edge_dist=-1)
my_layout = g2.layout_drl(weights=1/np.asarray(weights), fixed=None, seed=None, options={"dge_cut":0.7}, dim=3)
visual_style["layout"] = my_layout# Plot the graph
#visual_style["layout"] = "circle"
#visual_style["layout"] = "random"
#visual_style["layout"] = "star"
fig2, ax2 = plt.subplots()
visual_style["target"] = ax2
#ig.plot(g, out_name, **visual_style)
ig.plot(g2, **visual_style)
plt.show(block=False)






###############################################################################



import matplotlib.pyplot as plt
import networkx as nx


# create a directed multi-graph
G = nx.MultiDiGraph()
G.add_edges_from(edgeList)
# plot the graph
plt.figure(figsize=(8,8))
nx.draw(G, connectionstyle='arc3, rad = 0.1')
plt.show(block=False)  # pause before exiting



