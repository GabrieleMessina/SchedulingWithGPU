#include "dag.h"
#include "utils.h"

//TODO: eliminare??

template<typename  T>
class GraphVector : public Graph<T> {
private:
	vector<edge_t> adj_vec; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
public:
	GraphVector(int len = 100) : Graph<T>(len)
	{
	}

	~GraphVector(){
		//not needed if adj is a std::vector
	}

	void initAdjacencyMatrix() override {
		Graph<T>::adj_len = Graph<T>::len * Graph<T>::len;
		adj_vec.reserve(Graph<T>::adj_len);
		adj_vec.resize(Graph<T>::adj_len);
		printf("GraphVector: len is %d and mat is %d\n", Graph<T>::len, Graph<T>::adj_len);
	}

	Graph<T>* insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1) override {
		if (adj_vec.size() == 0) initAdjacencyMatrix();

		int i = indexOfa;
		int j = indexOfb;
		if (i > -1 && j > -1 && i < Graph<T>::len && j < Graph<T>::len) {
#if TRANSPOSED_ADJ
			int matrixToArrayIndex = matrix_to_array_indexes(j, i, Graph<T>::len);
#else
			int matrixToArrayIndex = matrix_to_array_indexes(i, j, Graph<T>::len);
#endif

			adj_vec.insert(adj_vec.begin() + matrixToArrayIndex, weight);
			Graph<T>::m++;
		}
		else {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
		return this;
	}

	edge_t* GetWeightsArray() override {
		return adj_vec.data();
	}
	
	edge_t* GetEdgesArray() override {
		return adj_vec.data();
	}

	bool hasEdge(T a, T b) override {
		int i = indexOfNode(a);
		int j = indexOfNode(b);
		return hasEdgeByIndex(i, j);
	}

	bool hasEdgeByIndex(int indexOfa, int indexOfb) override {
		int i = indexOfa;
		int j = indexOfb;
		if (i != -1 && j != -1) {
#if TRANSPOSED_ADJ
			int matrixToArrayIndex = matrix_to_array_indexes(j, i, Graph<T>::len);
#else
			int matrixToArrayIndex = matrix_to_array_indexes(i, j, Graph<T>::len);
#endif

			return adj_vec.at(matrixToArrayIndex) != 0;
		}
		return false;
	}
};
