#include "dag.h"
#include "utils.h"

//TODO: modificare kernel per usare nuova struttura di adj.

template<typename  T>
class GraphRectangular : public Graph<T> {
private: 
	edge_t* adj = NULL; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
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
#pragma unroll
		for (int i = 0; i < Graph<T>::adj_len; i++) {
			adj[i] = emptyAdjCell;
		}
		printf("GraphRectangular: len is %d and mat is %d\n", Graph<T>::len, Graph<T>::adj_len);
	}

	bool hasEdge(T a, T b) override {
		int i = Graph<T>::indexOfNode(a);
		int j = Graph<T>::indexOfNode(b);
		return hasEdgeByIndex(i, j);
	}

	bool hasEdgeByIndex(int indexOfa, int indexOfb) override {
		int i = indexOfa;
		int j = 0;
		if (i != -1 && j != -1) {
			int matrixToArrayIndex = -1;
			do {
#if TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j++, i, Graph<T>::len);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i, j++, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_len && adj[matrixToArrayIndex] != indexOfb);

			return matrixToArrayIndex < Graph<T>::adj_len; //se maggiore allora abbiamo percorso tutta la matrice senza trovare nulla
		}
		return false;
	}

	Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1) override {
		if (adj == NULL) initAdjacencyMatrix();

		int i = indexOfa;
		int j = 0;
		if (i > -1 && j > -1 && i < Graph<T>::len && j < Graph<T>::len) {
			int matrixToArrayIndex = -1;
			do {
#if TRANSPOSED_ADJ
				matrixToArrayIndex = matrix_to_array_indexes(j++, i, Graph<T>::len);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i, j++, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_len && adj[matrixToArrayIndex] > emptyAdjCell);

			adj[matrixToArrayIndex] = indexOfb;
			Graph<T>::m++;
		}
		else {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
		return this;
	}

	edge_t* GetEdgesArray() override {
		return adj;
	}
};