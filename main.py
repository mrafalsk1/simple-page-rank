import numpy as np

def page_rank(graph, damping = 0.85, iterations=1000, tolerance= 0.000001):
    pages: int = graph.shape[0];
    rank = np.ones(pages) / pages;
    column_sums = np.sum(graph, axis=0);
    column_sums[column_sums == 0] = 1; 
    transition_matrix = graph / column_sums;
    for i in range(iterations):
        new_rank = (1 - damping) / pages + damping * np.dot(transition_matrix, rank);
        if np.linalg.norm(new_rank - rank, 1) < tolerance:
            break

        rank = new_rank;

    return rank;
if __name__ == "__main__":
    graph = np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 1, 0]
    ]) 

    print(graph)
    rank = page_rank(graph);
    print(rank)
