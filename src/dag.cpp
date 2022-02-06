#include "dag.h"
#include "utils.h"
#include "dag_vector.cpp"
#include "dag_rectangular.cpp"

edge_t* adj = NULL; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.

template<class T>
Graph<T>::Graph(int len) {
	max_edges_for_node = n = m = 0;
	this->len = len;
	adj_len = len * len;
	nodes = DBG_NEW T[len];
	for (int i = 0; i < len; i++) nodes[i] = 0;
	adj = NULL;
}

template<class T>
Graph<T>::~Graph() {
	delete[] nodes;
	if(adj != NULL) delete[] adj; 
}

template<class T>
int Graph<T>::insertNode(T key) {
	if (n < len) {
		nodes[n] = key;
	}
	return n++;
}

template<class T>
int Graph<T>::insertUniqueValueNode(T key) {
	if (indexOfNode(key) != -1) return -1;
	if (n < len) {
		nodes[n] = key;
	}
	return n++;
}

template<class T>
int Graph<T>::indexOfNode(T key) {
	for (int i = 0; i < len; i++) if (nodes[i] == key) return i;
	return -1;
}

template<class T>
void Graph<T>::initAdjacencyMatrix() {
	adj_len = len * len;
	adj = DBG_NEW edge_t[adj_len];
#pragma unroll
	for (int i = 0; i < adj_len; i++) {
		adj[i] = 0;
	}
	printf("Graph<T>: len is %d and mat is %d\n", len, adj_len);
}

template<class T>
Graph<T>* Graph<T>::insertEdge(T a, T b, int weight) {
	//TODO: verifica che non crei cicli!
	int i = indexOfNode(a);
	int j = indexOfNode(b);

	if (i > -1 && j > -1 && i < len && j < len) {
		if (typeid(int) == typeid(a)) {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", a, b);
		}
		else if (typeid(string) == typeid(b)) {
			printf("impossibile aggiungere l'edge perche' uno dei valori non esiste in insertEdge(%d,%d)\n", a, b);
		}
		return this;
	}
	
	return insertEdgeByIndex(a, b, weight);
}

template<class T>
Graph<T>* Graph<T>::insertEdgeByIndex(int indexOfa, int indexOfb, int weight) {
	if (adj == NULL) initAdjacencyMatrix();

	int i = indexOfa;
	int j = indexOfb;
	if (i > -1 && j > -1 && i < len && j < len) {
		int matrixToArrayIndex;
#if TRANSPOSED_ADJ
		matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#else
		matrixToArrayIndex = matrix_to_array_indexes(i, j, len);
#endif // TRANSPOSED_ADJ

		adj[matrixToArrayIndex] = weight;
		m++;
	}
	else {
		printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
	}
	return this;
}

template<class T>
edge_t* Graph<T>::GetEdgesArray(){
	return adj;
}

template<class T>
edge_t* Graph<T>::GetEdgesReverseArray(){
	error("Not implemented Exception");
	return NULL;
}

template<class T>
bool Graph<T>::hasEdge(T a, T b){
	int i = indexOfNode(a);
	int j = indexOfNode(b);
	return hasEdgeByIndex(i, j);
}

template<class T>
int Graph<T>::numberOfParentOfNode(int indexOfNode){
	error("Not implemented Exception");
	return 0;
}
template<class T>
int Graph<T>::numberOfChildOfNode(int indexOfNode) {
	error("Not implemented Exception");
	return 0;
}

template<class T>
bool Graph<T>::hasEdgeByIndex(int indexOfa, int indexOfb){
	int i = indexOfa;
	int j = indexOfb;
	if (i != -1 && j != -1) {
#if TRANSPOSED_ADJ
		int matrixToArrayIndex = matrix_to_array_indexes(j, i, len);
#else
		int matrixToArrayIndex = matrix_to_array_indexes(i, j, len);
#endif

		return adj[matrixToArrayIndex] != 0;
	}
	return false;
}



template <>
Graph<int>* Graph<int>::initDagWithDataSet(const char* dataset_file_name) {
	ifstream data_set;
	stringstream ss;
	ss << "./data_set/" << dataset_file_name << ".txt";
	string dataset_file_name_with_extension = ss.str();
	data_set.open(dataset_file_name_with_extension);
	cout << "dataset_file_name_with_extension: " << dataset_file_name_with_extension << endl;
	if (!data_set.is_open()) {
		fprintf(stderr, "%s\n", "impossibile aprire il dataset");
		exit(1);
	}

	int n_nodes = 0;
	data_set >> n_nodes;

#if VECTOR_ADJ
	Graph<int>* DAG = DBG_NEW GraphVector<int>(n_nodes);
#elif RECTANGULAR_ADJ
	Graph<int>* DAG = DBG_NEW GraphRectangular<int>(n_nodes);
#else
	Graph<int>* DAG = DBG_NEW Graph<int>(n_nodes);
#endif

	if (!DAG) {
		data_set.close();
		fprintf(stderr, "%s\n", "failed to allocate graph");
		exit(1);
	}

	//leggo tutti gli id in prima posizione in modo da creare la dag senza adj per il momento.
	DAG->max_edges_for_node = 0; //TODO: va bene per il numero massimo di figli che è fisso, ma non è fisso il numero di padri.
	int value, n_successor, successor_index, data_transfer;
	while (data_set >> value >> n_successor) {
		int current_node_index = DAG->insertNode(value);
		DAG->max_edges_for_node = max(DAG->max_edges_for_node, n_successor);
		for (int i = 0; i < n_successor; i++)
		{
			data_set >> successor_index >> data_transfer;
		}
	}

	data_set.clear();
	data_set.seekg(0);
	data_set >> n_nodes;

	//adesso leggo di nuovo per creare la adj
	int current_node_index = 0;
	while (data_set >> value >> n_successor) {
		for (int i = 0; i < n_successor; i++)
		{
			data_set >> successor_index >> data_transfer;
			if (successor_index <= current_node_index) {
				cerr << "si sta tentando di aggiungere un arco che potrebbe causare un loop e rendere quindi il grafico ciclico" << endl;
				continue;
			}
			//questo è più veloce perché crea meno nodi in quanto nodi con lo stesso peso vengono considerati lo stesso nodo: DAG->insertEdgeByIndex(DAG->indexOfNode(value), successor_index, 1/*data_transfer*/); //assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
			DAG->insertEdgeByIndex(current_node_index, successor_index, 1/*data_transfer*/); //assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
		}
		current_node_index++;
	}

	cout << "#edge:" << DAG->m << endl;

	//padding per convenienza
	while (current_node_index < n_nodes)
	{
		DAG->insertNode(-1);
		current_node_index++;
	}

	printf("DAG initialized\n");
	if (DEBUG_DAG_INIT) {
		DAG->Print();
		cout << "\n";
		//print(DAG->adj, DAG->len, DAG->len, ", ");
	}
	cout << endl;

	data_set.close();
	return DAG;
}