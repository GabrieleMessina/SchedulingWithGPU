
/* IMPLEMENTAZIONE DI UNA DAG (DIRECT ACYCLIC GRAPH)*/
#pragma once
#include "app_globals.h"
#include "utils.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>

template<typename  T> 
class Graph {
public:
	//lunghezza approssimata ad un numero comodo
	int len;
	//superficie della matrice
	int adj_len;
	//numero max di edge per ogni nodo
	int max_edges_for_node;
	//numero reale di nodi(minore di len che è approssimato) 
	int n;
	//numero di archi
	int m;
	T* nodes;

	Graph(int len = 100);
	virtual ~Graph();

	virtual int insertNode(T key);

	virtual int insertUniqueValueNode(T key);

	virtual int indexOfNode(T key);
	
	virtual void initAdjacencyMatrix();

	virtual Graph<T>* insertEdge(T a, T b, int weight = 1);
	
	virtual Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1);

	virtual edge_t* GetEdgesArray();

	virtual bool hasEdge(T a, T b);
	
	virtual bool hasEdgeByIndex(int indexOfa, int indexOfb);

	//TODO: rendere virtuali? o riescono a chiamare fun più ad alto livello?
	void Print(){
		cout << "(index, value) -> [(edges_index, weight)]" << endl;
		int matrixToArrayIndex;
		edge_t* edges = GetEdgesArray();
		for (int i = 0; i < len; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
			for (int j = 0; j < len; j++) {
#if TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i, j, len);
#endif // TRANSPOSED_ADJ
				if (edges[matrixToArrayIndex] != 0) cout << "(" << j << ", " << edges[matrixToArrayIndex] << ") | ";
			}
			cout << endl;
		}
	}
	
	static Graph<int>* initDagWithDataSet(const char* dataset_file_name);
};

