from flask import Flask, request, jsonify
import networkx as nx
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

    # Gete PageRank
    pagerank_scores = nx.pagerank(subgraph, alpha=alpha)

    # Return top 10
    top_papers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    response = [
        {"paper_id": paper_id, "score": score} for paper_id, score in top_papers
    ]

    return jsonify({"success": True, "top_papers": response})


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

    # Get HITS scores
    hub_scores, authority_scores = nx.hits(subgraph)

    # Get top 10
    top_authority = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
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
