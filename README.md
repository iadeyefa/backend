# Citation Graph Backend

This project is a backend service built with Flask that provides endpoints for searching and retrieving citation data from a citation graph. It utilizes various algorithms such as PageRank and HITS to rank papers based on their citations.

## Requirements

- Python 3.6 or higher
- Flask
- Flask-CORS
- NetworkX
- NumPy
- Pickle

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install Flask Flask-CORS networkx numpy
   ```

4. **Download the Dataset:**
   - Download the citation dataset from [AMiner](https://www.aminer.cn/citation).
   - After downloading, ensure that the dataset is processed and saved as `citation_graph.pkl` in the `outputs` directory.

## Running the Application

To run the application, execute the following command:

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000/` by default.

## API Endpoints

### Home

- **GET** `/`

  Returns a simple message indicating that the home page is working.

### Search

- **GET** `/search`

  This endpoint allows you to search for papers based on a query. You can also specify various weights for the ranking algorithms.

  **Query Parameters:**
  - `query`: The search term to look for in paper titles.
  - `number_of_results`: The number of results to return (default is 10).
  - `salsa`: Weight for the SALSA algorithm (default is 0). - TODO
  - `hits`: Weight for the HITS algorithm (default is 1).
  - `hits_hub`: Weight for HITS hub score (default is 1).
  - `hits_authority`: Weight for HITS authority score (default is 1).
  - `pagerank`: Weight for the PageRank algorithm (default is 1).
  - `semantic_similarity`: Weight for semantic similarity (default is 0). - TODO
  - `publish_date`: Weight for publish date (default is 0). - TODO

  **Response:**
  Returns a JSON object containing the search results and the subgraph of citations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [NetworkX](https://networkx.org/)
- [NumPy](https://numpy.org/)