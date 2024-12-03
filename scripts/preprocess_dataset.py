import networkx as nx
import pickle

def preprocess_dataset(input_file, output_file, max_papers=100000):
    citation_graph = nx.DiGraph()

    with open(input_file, "r", encoding="utf-8") as file:
        current_paper = {}
        count = 0

        for line in file:
            if line.startswith("#*"):  # Paper title
                current_paper["title"] = line[2:].strip()
            elif line.startswith("#@"):  # Authors
                current_paper["authors"] = line[2:].strip()
            elif line.startswith("#t"):  # Year
                current_paper["year"] = line[2:].strip()
            elif line.startswith("#c"):  # Venue
                current_paper["venue"] = line[2:].strip()
            elif line.startswith("#index"):  # Paper ID
                current_paper["id"] = line[6:].strip()
            elif line.startswith("#%"):  # References
                if "references" not in current_paper:
                    current_paper["references"] = []
                current_paper["references"].append(line[2:].strip())
            elif line.strip() == "":  # End of record
                if "id" in current_paper:  # Add paper and references to graph
                    paper_id = current_paper["id"]
                    citation_graph.add_node(
                        paper_id,
                        title=current_paper.get("title", ""),
                        authors=current_paper.get("authors", ""),
                        year=current_paper.get("year", ""),
                        venue=current_paper.get("venue", ""),
                    )
                    for ref in current_paper.get("references", []):
                        citation_graph.add_edge(paper_id, ref)

                current_paper = {}
                count += 1
                if count >= max_papers:
                    break

    # Save the graph using pickle
    with open(output_file, "wb") as f:
        pickle.dump(citation_graph, f)
    print(f"Citation graph saved to {output_file} with {len(citation_graph.nodes)} nodes.")

# Preprocess dataset
preprocess_dataset("data/outputacm.txt", "outputs/citation_graph.pkl", max_papers=100000)
