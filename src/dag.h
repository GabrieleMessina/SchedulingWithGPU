
/* IMPLEMENTAZIONE DI UNA DAG (DIRECT ACYCLIC GRAPH)*/
#pragma once
#include "app_globals.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include "utils.h"

template<typename  T> 
class Graph {
public:
	//lunghezza approssimata ad un numero comodo
	int len;
	//superficie della matrice
	int adj_len;
	//superficie della matrice dei parent
	int adj_reverse_len;
	//numero max di edge per ogni nodo
	int max_children_for_nodes;
	//numero max di parent per ogni nodo
	int max_parents_for_nodes;
	//numero di processori
	int number_of_processors;
	//lunghezza della matrice dei costi per ogni processore
	int cost_on_processors_lenght;
	//numero reale di nodi(minore di len che è approssimato) 
	int n;
	//numero di archi
	int m;
	T* nodes;

	Graph(int len = 100, int processor_count = 1);
	virtual ~Graph();

	virtual int insertNode(T key);

	virtual int insertUniqueValueNode(T key);

	virtual int indexOfNode(T key);
	
	virtual void initAdjacencyMatrix();

	virtual Graph<T>* insertEdge(T a, T b, int weight = 1);
	
	virtual Graph<T>* insertCostForProcessor(int indexOfNode, int indexOfProcessor, int cost = 1);
	
	virtual Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1);

	virtual int* GetCostsArray();

	virtual edge_t* GetWeightsArray();

	virtual edge_t* GetWeightsReverseArray();
	
	virtual edge_t* GetEdgesArray();

	virtual edge_t* GetEdgesReverseArray();

	virtual bool hasEdge(T a, T b);
	
	virtual bool hasEdgeByIndex(int indexOfa, int indexOfb);
	
	virtual int numberOfParentOfNode(int indexOfNode);
	
	virtual int numberOfChildOfNode(int indexOfNode);

	void PrintCosts() {
		cout << "adj dei costi per nodo e processore: \n" << endl;
		cout << "(index, value) -> [(processor_index, cost)]" << endl;
		int* cost_on_processors = GetCostsArray();
		int matrixToArrayIndex;
		for (int i = 0; i < len; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
			for (int j = 0; j < number_of_processors; j++) {
				matrixToArrayIndex = matrix_to_array_indexes(i, j, number_of_processors);
				cout << "(" << j << ", " << cost_on_processors[matrixToArrayIndex] << ") | ";
			}
			cout << endl;
		}
	}

	void PrintReverse() {
		cout << "adj dei figli per nodo: \n" << endl;
		cout << "(index, value) -> [(edges_index, weight)]" << endl;
		int matrixToArrayIndex;
		edge_t* edges = GetEdgesReverseArray();
		edge_t* weights = GetWeightsReverseArray();
		for (int i = 0; i < len; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
#if RECTANGULAR_ADJ
			for (int j = 0; j < max_children_for_nodes; j++) {
#else
			for (int j = 0; j < len; j++) {
#endif
#if RECTANGULAR_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#elif TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i, j, len);
#endif // TRANSPOSED_ADJ
				/*if (edges[matrixToArrayIndex] != -1) */cout << "(" << j << ", " << edges[matrixToArrayIndex] << " w: " << weights[matrixToArrayIndex] << ") | ";
			}
			cout << endl;
		}
	}

	//TODO: rendere virtuali? o riescono a chiamare fun più ad alto livello?
	void Print(){
		cout << "adj dei parent per nodo: \n" << endl;
		cout << "(index, value) -> [(edges_index, weight)]" << endl;
		int matrixToArrayIndex;
		edge_t* edges = GetEdgesArray();
		edge_t* weights = GetWeightsArray();
		for (int i = 0; i < len; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
#if RECTANGULAR_ADJ
			for (int j = 0; j < max_parents_for_nodes; j++) {
#else
			for (int j = 0; j < len; j++) {
#endif
#if RECTANGULAR_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#elif TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i, j, len);
#endif // TRANSPOSED_ADJ
				/*if (edges[matrixToArrayIndex] != -1)*/ cout << "(" << j << ", " << edges[matrixToArrayIndex] << " w: "<< weights[matrixToArrayIndex] << ") | ";
			}
			cout << endl;
		}

#if RECTANGULAR_ADJ
		PrintReverse();
#endif
		PrintCosts();
	}
	
	static Graph<int>* initDagWithDataSet(const char* dataset_file_name);
};

