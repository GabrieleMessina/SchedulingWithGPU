
/* IMPLEMENTAZIONE DI UNA DAG (DIRECT ACYCLIC GRAPH)*/
#pragma once
#include <iostream>
#include "utils.h"

template<class T> class Graph {
public:
	//lunghezza approssimata ad un numero comodo, superficie della matrice, numero reale di nodi(minore di len che è approssimato) ,numero di archi
	int len, adj_len, n, m;
	T* nodes;
	bool* adj; //TODO: adj potrebbe già essere un array in modo da passarlo direttamente alla GPU senza doverlo convertire.

	Graph(int len = 100);
	~Graph();

	int insertNode(T key);

	int insertUniqueValueNode(T key);

	int indexOfNode(T key);

	Graph<T>* insertEdge(T a, T b, int weight = 1);

	Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1);

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

