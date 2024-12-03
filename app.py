from flask import Flask, request, jsonify
import networkx as nx
import pickle

app = Flask(__name__)

with open("outputs/citation_graph.pkl", "rb") as fi:
    citation_graph = pickle.load(fi)

@app.route("/")
def home():
    return "Home Page Working"

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "").lower()
    max_results = int(request.args.get("max_results", 10))

    matching_papers = [
        node for node, data in citation_graph.nodes(data=True)
        if query in data.get("title", "").lower()
    ]

    response = [{"paper_id": paper_id, "title": citation_graph.nodes[paper_id]["title"]} for paper_id in
                matching_papers[:max_results]]

    return jsonify({"success": True, "results": response})

@app.route("/pagerank", methods=["GET"])
def pagerank():
    query = request.args.get("query", "").lower()
    alpha = float(request.args.get("alpha", 0.85))

    # Filter graph
    matching_papers = [
        node for node, data in citation_graph.nodes(data=True)
        if query in data.get("title", "").lower()
    ]
    subgraph = citation_graph.subgraph(matching_papers)

    # Gete PageRank
    pagerank_scores = nx.pagerank(subgraph, alpha=alpha)

    # Return top 10
    top_papers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    response = [{"paper_id": paper_id, "score": score} for paper_id, score in top_papers]

    return jsonify({"success": True, "top_papers": response})

@app.route("/hits", methods=["GET"])
def hits():
    query = request.args.get("query", "").lower()

    # Filter graph
    matching_papers = [
        node for node, data in citation_graph.nodes(data=True)
        if query in data.get("title", "").lower()
    ]
    subgraph = citation_graph.subgraph(matching_papers)

    # Get HITS scores
    hub_scores, authority_scores = nx.hits(subgraph)

    # Get top 10
    top_authority = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_hub = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    response = {
        "top_authority": [{"paper_id": paper_id, "score": score} for paper_id, score in top_authority],
        "top_hub": [{"paper_id": paper_id, "score": score} for paper_id, score in top_hub],
    }

    return jsonify({"success": True, "scores": response})

if __name__ == "__main__":
    app.run(debug=True)
