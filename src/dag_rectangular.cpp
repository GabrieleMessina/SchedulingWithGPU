#include "dag.h"
#include "utils.h"

//TODO: modificare kernel per usare nuova struttura di adj.

template<typename  T>
class GraphRectangular : public Graph<T> {
private: 
	edge_t* adj = NULL; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
	edge_t* adj_reverse = NULL; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
	int emptyAdjCell = -1;
public:

	GraphRectangular(int len = 100) : Graph<T>(len)
	{
	}

	~GraphRectangular(){
		delete[] adj;
	}

	void initAdjacencyMatrix() override {
		Graph<T>::adj_len = Graph<T>::len * Graph<T>::max_edges_for_node;
		adj = DBG_NEW edge_t[Graph<T>::adj_len];
		adj_reverse = DBG_NEW edge_t[Graph<T>::adj_len];
#pragma unroll
		for (int i = 0; i < Graph<T>::adj_len; i++) {
			adj[i] = emptyAdjCell;
			adj_reverse[i] = emptyAdjCell;
		}
		printf("GraphRectangular: len is %d and mat is %d with dept %d\n", Graph<T>::len, Graph<T>::adj_len, Graph<T>::max_edges_for_node);
	}

	bool hasEdge(T a, T b) override {
		int i = Graph<T>::indexOfNode(a);
		int j = Graph<T>::indexOfNode(b);
		return hasEdgeByIndex(i, j);
	}

	bool hasEdgeByIndex(int indexOfa, int indexOfb) override {
		int i = 0;
		int j = indexOfb;
		if (i != -1 && j != -1) {
			int matrixToArrayIndex = -1;
			do {
#if TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i++, Graph<T>::max_edges_for_node);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i++, j, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_len && adj[matrixToArrayIndex] != indexOfa && adj[matrixToArrayIndex] != emptyAdjCell);

			//se maggiore allora abbiamo percorso tutta la matrice senza trovare nulla
			if (matrixToArrayIndex < Graph<T>::adj_len) {
				return adj[matrixToArrayIndex] == indexOfa;
			}
			else return false;

		}
		return false;
	}

	Graph<T>* insertEdgeByIndexReverse(int indexOfa, int indexOfb, int weight = 1) {
		int i = indexOfa;
		int j = 0;
		if (indexOfa > -1 && indexOfa > -1 && indexOfa < Graph<T>::len && indexOfb < Graph<T>::len) {
			int matrixToArrayIndex = -1;
			do {
#if TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(i, j++, Graph<T>::max_edges_for_node);
#else
				matrixToArrayIndex = matrix_to_array_indexes(j++, i, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_len && adj_reverse[matrixToArrayIndex] > emptyAdjCell);

			if (matrixToArrayIndex >= Graph<T>::adj_len) {
				printf("ERROR: si è cercato di inserire un edge oltre il limite di profondità di una DAG con matrice adj rettangolare.");
			}
			else adj_reverse[matrixToArrayIndex] = indexOfb;
		}
		else {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
		return this;
	}

	Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1) override {
		if (adj == NULL) initAdjacencyMatrix();

		int i = 0;
		int j = indexOfb;
		if (indexOfa > -1 && indexOfa > -1 && indexOfa < Graph<T>::len && indexOfb < Graph<T>::len) {
			int matrixToArrayIndex = -1;
			do {
#if TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j, i++, Graph<T>::max_edges_for_node);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i++, j, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_len && adj[matrixToArrayIndex] > emptyAdjCell);

			if (matrixToArrayIndex >= Graph<T>::adj_len) {
				printf("ERROR: si è cercato di inserire un edge oltre il limite di profondità di una DAG con matrice adj rettangolare.\n");
			}
			else adj[matrixToArrayIndex] = indexOfa;
			Graph<T>::m++;
		}
		else {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
		insertEdgeByIndexReverse(indexOfa, indexOfb, weight);
		return this;
	}

	edge_t* GetEdgesArray() override {
		return adj;
	}
	edge_t* GetEdgesReverseArray() override {
		return adj_reverse;
	}

	int numberOfParentOfNode(int indexOfNode) override {
		if (indexOfNode < -1 && indexOfNode >= Graph<T>::len) return 0;
		int parentCount = -1;
		int matrixToArrayIndex;
		int parent;
		do {
			matrixToArrayIndex = matrix_to_array_indexes(++parentCount, indexOfNode, Graph<T>::len);
			if (matrixToArrayIndex >= Graph<T>::adj_len) return parentCount;
			parent = adj[matrixToArrayIndex];
		} while (parent > -1 && parentCount < Graph<T>::max_edges_for_node);
		return parentCount;
	}
	int numberOfChildOfNode(int indexOfNode) override {
		if (indexOfNode < -1 && indexOfNode >= Graph<T>::len) return 0;
		int childCount = -1;
		int matrixToArrayIndex;
		int child;
		do {
			matrixToArrayIndex = matrix_to_array_indexes(++childCount, indexOfNode, Graph<T>::len);
			if (matrixToArrayIndex >= Graph<T>::adj_len) return childCount;
			child = adj_reverse[matrixToArrayIndex];
		} while (child > -1 && childCount < Graph<T>::max_edges_for_node);
		return childCount;
	}
};