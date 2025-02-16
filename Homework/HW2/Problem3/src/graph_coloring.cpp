// This program involves coloring a graph G(V, E) such that no two adjacent vertices have the same color.
// The goal is to use the least number of colors to color the graph, where an exact solution to graph coloring is NP-hard.
// Author: Zander Ingare

#include <omp.h>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <chrono>
#include <algorithm>
#include <iomanip>

#define NUM_THREADS 4
#define NUM_VERTICES 10

// Generates an adjacency list where each vertex stores its neighbors
std::map<int, std::set<int>> generate_graph(int graph_size) {
    std::map<int, std::set<int>> graph;

    // Populate the graph with empty values
    for (int i = 0; i < graph_size; i++) {
        graph[i] = std::set<int>();
    }

    // Loop over the vertices in the graph
    for (int i = 0; i < graph_size; i++) {
        for (int j = 0; j < graph_size; j++) {
            if (i != j) {
                // Random generation of neighbors for the vertex
                if (rand() % 2 == 0) {
                    graph[i].insert(j);
                    graph[j].insert(i); // Ensure the neighbor relationship is bidirectional
                }
            }
        }
    }
    return graph;
}

// Colors the graph using a sequential algorithm
std::map<int, int> color_vertices_sequential(const std::map<int, std::set<int>> &graph) {
    std::map<int, int> colored_vertices; // A map of the vertex and its assigned color value
    std::map<int, std::set<int>> unavailable_colors_map; // A dictionary of unavailable colors for each vertex

    for (int v = 0; v < NUM_VERTICES; v++) {
        int color = 0;

        // Check the unavailable colors for the current vertex
        if (unavailable_colors_map.find(v) != unavailable_colors_map.end()) {        
            while (unavailable_colors_map[v].find(color) != unavailable_colors_map[v].end()) {
                color++;
            }
        }

        colored_vertices[v] = color;

        // Mark the current vertex color as unavailable for all of its neighbors (if they exist)
        if (graph.find(v) != graph.end()) {
            for (int neighbor : graph.at(v)) {
                unavailable_colors_map[neighbor].insert(color);
            }
        }
    }

    return colored_vertices;
}

// Colors the graph using the Jones-Plassmann algorithm (parallel) and OpenMP
std::map<int, int> color_vertices_parallel(const std::map<int, std::set<int>> &graph) {
    int n = graph.size();
    std::vector<int> priorities(n);
    std::vector<bool> colored(n, false);
    std::map<int, int> colored_graph;  // Resultant colored graph (vertex -> color)

    // Assign random priorities to each vertex
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; i++) {
        priorities[i] = rand();
    }

    std::vector<int> uncolored_nodes;
    for (const auto &entry : graph) {
        uncolored_nodes.push_back(entry.first);
    }

    while (!uncolored_nodes.empty()) {
        std::vector<int> to_color;

        // Select nodes that have the highest priority in their neighborhood
        #pragma omp parallel
        {
            std::vector<int> local_to_color;

            #pragma omp for nowait
            for (size_t i = 0; i < uncolored_nodes.size(); i++) {
                int node = uncolored_nodes[i];
                bool highest_priority = true;

                for (int neighbor : graph.at(node)) {
                    if (!colored[neighbor] && priorities[neighbor] > priorities[node]) {
                        highest_priority = false;
                        break;
                    }
                }

                if (highest_priority) {
                    local_to_color.push_back(node);
                }
            }

            #pragma omp critical
            to_color.insert(to_color.end(), local_to_color.begin(), local_to_color.end());
        }

        // Assign colors to selected nodes
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t i = 0; i < to_color.size(); i++) {
            int node = to_color[i];
            std::set<int> neighbor_colors;

            for (int neighbor : graph.at(node)) {
                if (colored[neighbor]) {
                    neighbor_colors.insert(colored_graph[neighbor]);
                }
            }

            int color = 1;
            while (neighbor_colors.find(color) != neighbor_colors.end()) {
                color++;
            }

            colored_graph[node] = color;
            colored[node] = true;
        }

        // Update uncolored nodes
        std::vector<int> new_uncolored_nodes;
        for (int node : uncolored_nodes) {
            if (!colored[node]) {
                new_uncolored_nodes.push_back(node);
            }
        }
        uncolored_nodes = std::move(new_uncolored_nodes);
    }

    return colored_graph;
}


// Validates that the colored graph is properly colored
bool validate_colored_vertices(const std::map<int, std::set<int>> &graph, const std::map<int, int> &colored_graph) {
    for (const auto &entry : graph) {
        int vertex = entry.first;
        for (int neighbor : entry.second) {
            if (colored_graph.at(vertex) == colored_graph.at(neighbor)) {
                std::cout << "Vertex " << vertex << " and Vertex " << neighbor << " have the same color!" << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Prints the adjacency list of the graph in a readable format
void print_graph(const std::map<int, std::set<int>> &graph) {
    std::cout << "Graph Adjacency Neighbors List:\n";
    for (const auto &entry : graph) {
        std::cout << "\t" << "Vertex " << entry.first << " --> ";
        for (int neighbor : entry.second) {
            std::cout << neighbor << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Prints the graph coloring results in a readable format
void print_colored_vertices(const std::map<int, int> &coloring) {
    std::cout << "Graph Coloring Results:\n";
    for (const auto &entry : coloring) {
        std::cout << "\t" << "Vertex " << entry.first << " --> Color " << entry.second << "\n";
    }
    std::cout << std::endl;
}

int main() {
    omp_set_num_threads(NUM_THREADS);
    srand(time(0)); // Seed for randomness

    // Generate a random graph and print it
    std::map<int, std::set<int>> graph = generate_graph(NUM_VERTICES);
    print_graph(graph);
    
    auto sequential_start_time = std::chrono::high_resolution_clock::now();
    std::map<int, int> colored_vertices_sequential = color_vertices_sequential(graph);
    auto sequential_end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> sequential_elapsed_time = sequential_end_time - sequential_start_time;

    // Print the assigned graph colors for the sequential algorithm
    print_colored_vertices(colored_vertices_sequential);

    // Validate the sequential algorithmically-colored graph
    if (validate_colored_vertices(graph, colored_vertices_sequential)) {
        std::cout << "Sequential algorithm graph coloring is valid!" << std::endl;
    } else {
        std::cout << "Sequential algorithm graph coloring is invalid!" << std::endl;
    }

    std::cout << "Sequential algorithm elapsed time: " << std::fixed << std::setprecision(15) << sequential_elapsed_time.count() << " seconds" << std::endl << std::endl;

    auto parallel_start_time = std::chrono::high_resolution_clock::now();
    std::map<int, int> colored_vertices_parallel = color_vertices_parallel(graph);
    auto parallel_end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> parallel_elapsed_time = parallel_end_time - parallel_start_time;

    // Print the assigned graph colors for the parallel algorithm
    print_colored_vertices(colored_vertices_parallel);

    // Validate the parallel algorithmically-colored graph
    if (validate_colored_vertices(graph, colored_vertices_parallel)) {
        std::cout << "Parallel algorithm graph coloring is valid!" << std::endl;
    } else {
        std::cout << "Parallel algorithm graph coloring is invalid!" << std::endl;
    }

    std::cout << "Parallel algorithm elapsed time: " << std::fixed << std::setprecision(15) << parallel_elapsed_time.count() << " seconds" << std::endl << std::endl;

    return 0;
}
