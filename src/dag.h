
/* IMPLEMENTAZIONE DI UNA DAG (DIRECT ACYCLIC GRAPH)*/
#pragma once
#include "app_globals.h"
#include "utils.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#define VECTOR_ADJ false

template<typename  T> 
class Graph {
public:
	//lunghezza approssimata ad un numero comodo, superficie della matrice, numero reale di nodi(minore di len che è approssimato) ,numero di archi
	int len, adj_len, n, m;
	T* nodes;
#if VECTOR_ADJ
	vector<edge_t> adj; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
#else
	edge_t* adj; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
#endif

	Graph(int len = 100);
	~Graph() {
		delete[] nodes;
		//delete[] adj; //not needed if adj is a std::vector
	}

	int insertNode(T key);

	int insertUniqueValueNode(T key);

	int indexOfNode(T key);

	Graph<T>* insertEdge(T a, T b, int weight = 1);
	
	Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1);

	edge_t* GetEdgesArray();

	bool hasEdge(T a, T b);

	void Print(){
		cout << "(index, value) -> [(edges_index, weight)]" << endl;
		for (int i = 0; i < len; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
			for (int j = 0; j < len; j++) {
				if (adj[matrix_to_array_indexes(i, j, len)] != 0) cout << "(" << j << ", " << adj[matrix_to_array_indexes(i, j, len)] << ") | ";
			}
			cout << endl;
		}
	}
	
	static Graph<int>* initDagWithDataSet(const char* dataset_file_name);
};

