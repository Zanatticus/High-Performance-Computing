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

#define NUM_THREADS 4
#define NUM_VERTICES 100

// Generates an adjacency list where each vertex stores its neighbors
std::map<int, std::vector<int>> generate_graph(int graph_size) {
    std::map<int, std::vector<int>> graph;

    // Loop over the vertices in the graph
    for (int i = 0; i < graph_size; i++) {
        std::vector<int> neighbors;
        for (int j = 0; j < graph_size; j++) {
            if (i != j) {
                // Random generation of neighbors for the vertex
                if (rand() % 2 == 0) {
                    neighbors.push_back(j);
                    graph[j].push_back(i); // Ensure the neighbor relationship is bidirectional
                }
            }
        }
        graph[i] = neighbors;
    }
    return graph;
}

// Colors the graph using a greedy algorithm and OpenMP
std::map<int, int> color_vertices(const std::map<int, std::vector<int>> &graph) {
    std::map<int, int> colored_vertices; // A map of the vertex and its assigned color value
    std::map<int, std::set<int>> unavailable_colors_map; // A dictionary of unavailable colors for each vertex
    
    for (int v = 0; v < NUM_VERTICES; v++) {
        int color = 0;

        // Check the unavailable colors for the current vertex
        if (unavailable_colors_map.find(v) != unavailable_colors_map.end()) {
            std::set<int> unavailable_colors = unavailable_colors_map[v];
            
            while (unavailable_colors.find(color) != unavailable_colors.end()) {
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

// Validates that the colored graph is properly colored
bool validate_colored_vertices(const std::map<int, std::vector<int>> &graph, const std::map<int, int> &colored_graph) {
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
void print_graph(const std::map<int, std::vector<int>> &graph) {
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
    std::map<int, std::vector<int>> graph = generate_graph(NUM_VERTICES);
    print_graph(graph);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    std::map<int, int> colored_vertices = color_vertices(graph);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Print the assigned graph colors
    print_colored_vertices(colored_vertices);

    // Validate the colored graph
    if (validate_colored_vertices(graph, colored_vertices)) {
        std::cout << "Graph coloring is valid!" << std::endl;
    } else {
        std::cout << "Graph coloring is invalid!" << std::endl;
    }

    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
