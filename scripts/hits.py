import numpy as np
import networkx as nx

def hits(graph, max_iter=100, tol=1e-6):
    nodes = list(graph.nodes())
    n = len(nodes)

    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)

    hubs = np.ones(n)
    authorities = np.ones(n)

    for _ in range(max_iter):
        new_authorities = adjacency_matrix.T @ hubs

        new_hubs = adjacency_matrix @ new_authorities

        new_authorities /= np.linalg.norm(new_authorities, ord=2)
        new_hubs /= np.linalg.norm(new_hubs, ord=2)

        if (np.allclose(hubs, new_hubs, atol=tol) and np.allclose(authorities, new_authorities, atol=tol)):
            break

        hubs = new_hubs
        authorities = new_authorities

    hub_scores = dict(zip(nodes, hubs))
    authority_scores = dict(zip(nodes, authorities))

    return hub_scores, authority_scores
