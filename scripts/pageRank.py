import networkx as nx
import numpy as np

def pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = list(graph.nodes())
    n = len(nodes)

    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)

    transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)
    transition_matrix[np.isnan(transition_matrix)] = 0  # Handle dangling nodes

    ranks = np.ones(n) / n

    for _ in range(max_iter):
        new_ranks = alpha * (transition_matrix @ ranks) + (1 - alpha) / n

        if np.allclose(ranks, new_ranks, atol=tol):
            break

        ranks = new_ranks

    pagerank_scores = dict(zip(nodes, ranks))

    return pagerank_scores
