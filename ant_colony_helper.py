import antcolony
from antgraph import AntGraph


class AntColonyHelper:
    # clust with ant colony optimization
    def ant_colony_optimization(n_users, pcs_matrix):
        graph = AntGraph(n_users, pcs_matrix)
        graph.reset_tau()
        num_iterations = 5
        # n_users = 5
        ant_colony = antcolony.AntColony(graph, 5, num_iterations)
        ant_colony.start()
        # graph.delta_mat = is_max(graph.delta_mat)
        # result = connected_component_labelling(graph.delta_mat, 4)
        return graph.delta_mat
