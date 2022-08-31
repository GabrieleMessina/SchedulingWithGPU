#include "dag.h"
#include "utils.h"


template<typename  T>
class GraphRectangular : public Graph<T> {
private: 
	edge_t* weights = NULL; // è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
	edge_t* adj = NULL; //è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
	edge_t* adj_reverse = NULL; //è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
	int emptyAdjCell = -1;
public:

	GraphRectangular(int len = 100, int processor_count = 1) : Graph<T>(len, processor_count)
	{
	}

	~GraphRectangular(){
		delete[] adj;
		delete[] adj_reverse;
		delete[] weights;
	}

	void initAdjacencyMatrix() override {
		//in adj troviamo i parent per ogni nodo, in reverse troviamo i child di ogni nodo,
		//inoltre la matrice può essere vista come una hashmap, quindi ogni colonna possiede i parent o i child di ogni nodo, quindi ha n_nodes colonne e max_adj righe.
		Graph<T>::adj_len = Graph<T>::len * Graph<T>:: max_parents_for_nodes;
		Graph<T>::adj_reverse_len = Graph<T>::len * Graph<T>::max_children_for_nodes;
		adj = DBG_NEW edge_t[Graph<T>::adj_len];
		weights = DBG_NEW edge_t[Graph<T>::adj_len];
		adj_reverse = DBG_NEW edge_t[Graph<T>::adj_reverse_len];
#pragma unroll
		for (int i = 0; i < Graph<T>::adj_len; i++) {
			adj[i] = emptyAdjCell;
			weights[i] = 0;
		}
#pragma unroll
		for (int i = 0; i < Graph<T>::adj_reverse_len; i++) {
			adj_reverse[i] = emptyAdjCell;
		}
		printf("GraphRectangular: len is %d and mat is %d and rev_mat is %d with dept %d and parent dept %d\n", Graph<T>::len, Graph<T>::adj_len, Graph<T>::adj_reverse_len, Graph<T>::max_children_for_nodes, Graph<T>::max_parents_for_nodes);
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
				matrixToArrayIndex = matrix_to_array_indexes(j, i++, Graph<T>::max_children_for_nodes);
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
				matrixToArrayIndex = matrix_to_array_indexes(i, j++, Graph<T>::max_children_for_nodes);
#else
				matrixToArrayIndex = matrix_to_array_indexes(j++, i, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_reverse_len && adj_reverse[matrixToArrayIndex] > emptyAdjCell);

			if (matrixToArrayIndex >= Graph<T>::adj_reverse_len) {
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
				matrixToArrayIndex = matrix_to_array_indexes(j, i++, Graph<T>::max_parents_for_nodes);
#else
				matrixToArrayIndex = matrix_to_array_indexes(i++, j, Graph<T>::len);
#endif
			} while (matrixToArrayIndex < Graph<T>::adj_len && adj[matrixToArrayIndex] > emptyAdjCell);

			if (matrixToArrayIndex >= Graph<T>::adj_len) {
				printf("ERROR: si è cercato di inserire un edge oltre il limite di profondità di una DAG con matrice adj rettangolare.\n");
			}
			else {
				adj[matrixToArrayIndex] = indexOfa;
				weights[matrixToArrayIndex] = weight;
			}
			Graph<T>::m++;
		}
		else {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
		insertEdgeByIndexReverse(indexOfa, indexOfb, weight);
		return this;
	}


	edge_t* GetWeightsArray() override {
		return weights;
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
		} while (parent > -1 && parentCount < Graph<T>::max_parents_for_nodes);
		return parentCount;
	}

	int numberOfChildOfNode(int indexOfNode) override {
		if (indexOfNode < -1 && indexOfNode >= Graph<T>::len) return 0;
		int childCount = -1;
		int matrixToArrayIndex;
		int child;
		do {
			matrixToArrayIndex = matrix_to_array_indexes(++childCount, indexOfNode, Graph<T>::len);
			if (matrixToArrayIndex >= Graph<T>::adj_reverse_len) return childCount;
			child = adj_reverse[matrixToArrayIndex];
		} while (child > -1 && childCount < Graph<T>::max_children_for_nodes);
		return childCount;
	}
};