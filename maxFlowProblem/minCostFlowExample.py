#minCostFlowExample


import numpy as np
from ortools.graph.python import min_cost_flow


"""MinCostFlow simple interface example."""
# Instantiate a SimpleMinCostFlow solver.
smcf = min_cost_flow.SimpleMinCostFlow()

# Define four parallel arrays: sources, destinations, capacities,
# and unit costs between each pair. For instance, the arc from node 0
# to node 1 has a capacity of 15.
start_nodes = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4])
end_nodes = np.array([1, 2, 2, 3, 4, 3, 4, 4, 2])
capacities = np.array([15, 8, 20, 4, 10, 15, 4, 20, 5])
unit_costs = np.array([4, 4, 2, 2, 6, 1, 3, 2, 3])

# Define an array of supplies at each node.
supplies = [20, 0, 0, -5, -15]

# Add arcs, capacities and costs in bulk using numpy.
all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
    start_nodes, end_nodes, capacities, unit_costs)

# Add supply for each nodes.
for i in np.arange(len(supplies)):
    smcf.set_node_supply(i, supplies[i])

# Find the min cost flow.
status = smcf.solve()

if status != smcf.OPTIMAL:
    print('There was an issue with the min cost flow input.')
    print(f'Status: {status}')
    exit(1)
print(f'Minimum cost: {smcf.optimal_cost()}')
print('')
print(' Arc    Flow / Capacity Cost')
solution_flows = np.zeros(len(all_arcs))
for i in np.arange(len(all_arcs)):
    solution_flows[i] = smcf.flow(all_arcs[i])
solution_flows = solution_flows.tolist()

costs = solution_flows * unit_costs
for arc, flow, cost in zip(all_arcs, solution_flows, costs):
    print(
        f'{smcf.tail(arc):1} -> {smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}'
    )

