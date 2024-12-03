from flask import Flask, request, jsonify
import networkx as nx
import numpy as np
import pickle

app = Flask(__name__)

with open("outputs/citation_graph.pkl", "rb") as fi:
    citation_graph = pickle.load(fi)


@app.route("/")
def home():
    return "Home Page Working"


# uses min max normalization
def normalize_scores(scores):
    max_score = max(scores.values())
    min_score = min(scores.values())
    for node in scores:
        scores[node] = (scores[node] - min_score) / (max_score - min_score)
    return scores


# search endpoint with multiple weights
@app.route("/search", methods=["GET"])
def search():
    number_of_results = int(request.args.get("number_of_results", 10))
    salsa_weight = request.args.get("salsa", 0)
    hits_weight = request.args.get("hits", 1)
    hits_hub_weight = request.args.get("hits_hub", 1)
    hits_authority_weight = request.args.get("hits_authority", 1)
    pagerank_weight = request.args.get("pagerank", 1)
    eigenvector_weight = request.args.get("eigenvector", 0)
    semantic_similarity_weight = request.args.get("semantic_similarity", 0)
    publish_date_weight = request.args.get("publish_date", 0)

    hits_weight = hits_weight / 2
    hits_hub_weight = hits_hub_weight / 2
    hits_authority_weight = hits_authority_weight / 2

    pagerank_weight = pagerank_weight / 2

    hits_hub_scores, hits_authority_scores = nx.hits(citation_graph)
    pagerank_scores = nx.pagerank(citation_graph)

    hits_hub_scores = normalize_scores(hits_hub_scores)
    hits_authority_scores = normalize_scores(hits_authority_scores)
    pagerank_scores = normalize_scores(pagerank_scores)

    # O(n) time complexity
    overall_scores = {}
    for node in citation_graph.nodes():
        overall_scores[node] = (
            hits_weight
            * (
                hits_hub_weight * hits_hub_scores[node]
                + hits_authority_weight * hits_authority_scores[node]
            )
        ) + (pagerank_weight * pagerank_scores[node])
    overall_scores = normalize_scores(overall_scores)

    # O(n logn) time complexity
    top_papers = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)[
        :number_of_results
    ]

    response = []
    for paper in top_papers:
        node = paper[0]
        data = citation_graph.nodes[node]
        print(data)
        response.append(
            {
                "paper_id": node,
                "title": data.get("title", ""),
                "authors": data.get("authors", ""),
                "year": data.get("year", ""),
                "overall_score": paper[1],
            }
        )

    return jsonify({"success": True, "results": response})
    

def pagerank_helper(graph, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = list(graph.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Create adjacency matrix
    adjacency_matrix = np.zeros((n, n))
    for u, v in graph.edges():
        adjacency_matrix[node_idx[u], node_idx[v]] = 1

    # Handle dangling nodes
    out_degree = adjacency_matrix.sum(axis=1)
    for i in range(n):
        if out_degree[i] == 0:
            adjacency_matrix[i] = 1 / n

    # Normalize the adjacency matrix to create the transition matrix
    transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)

    ranks = np.ones(n) / n

    # Power iteration
    for _ in range(max_iter):
        new_ranks = alpha * (transition_matrix @ ranks) + (1 - alpha) / n
        if np.allclose(ranks, new_ranks, atol=tol):
            break
        ranks = new_ranks

    pagerank_scores = {node: ranks[i] for node, i in node_idx.items()}

    return pagerank_scores

@app.route("/pagerank", methods=["GET"])
def pagerank():
    query = request.args.get("query", "").lower()
    alpha = float(request.args.get("alpha", 0.85))

    # Filter graph
    matching_papers = [
        node
        for node, data in citation_graph.nodes(data=True)
        if query in data.get("title", "").lower()
    ]
    subgraph = citation_graph.subgraph(matching_papers)

    # Compute PageRank scores
    pagerank_scores = pagerank_helper(subgraph, alpha=alpha)

    # Get top 10 results
    top_papers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    response = [
        {"paper_id": paper_id, "score": score} for paper_id, score in top_papers
    ]

    return jsonify({"success": True, "top_papers": response})

def hits_helper(graph, max_iter=100, tol=1e-6):
    nodes = list(graph.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Create adjacency matrix
    adjacency_matrix = np.zeros((n, n))
    for u, v in graph.edges():
        adjacency_matrix[node_idx[u], node_idx[v]] = 1

    # hub and authority scores
    hubs = np.ones(n)
    authorities = np.ones(n)

    # Power iteration
    for _ in range(max_iter):
        # Update authority scores
        new_authorities = adjacency_matrix.T @ hubs
        # Update hub scores
        new_hubs = adjacency_matrix @ new_authorities

        # Normalize scores
        new_authorities /= np.linalg.norm(new_authorities, ord=2)
        new_hubs /= np.linalg.norm(new_hubs, ord=2)

        if np.allclose(hubs, new_hubs, atol=tol) and np.allclose(authorities, new_authorities, atol=tol):
            break

        hubs = new_hubs
        authorities = new_authorities

    hub_scores = {node: hubs[i] for node, i in node_idx.items()}
    authority_scores = {node: authorities[i] for node, i in node_idx.items()}

    return hub_scores, authority_scores

@app.route("/hits", methods=["GET"])
def hits():
    query = request.args.get("query", "").lower()

    # Filter graph
    matching_papers = [
        node
        for node, data in citation_graph.nodes(data=True)
        if query in data.get("title", "").lower()
    ]
    subgraph = citation_graph.subgraph(matching_papers)

    # Compute HITS scores
    hub_scores, authority_scores = hits_helper(subgraph)

    # Get top 10 results
    top_authority = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_hub = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    response = {
        "top_authority": [
            {"paper_id": paper_id, "score": score} for paper_id, score in top_authority
        ],
        "top_hub": [
            {"paper_id": paper_id, "score": score} for paper_id, score in top_hub
        ],
    }

    return jsonify({"success": True, "scores": response})


if __name__ == "__main__":
    app.run(debug=True)
