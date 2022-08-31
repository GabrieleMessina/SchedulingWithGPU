#include "dag.h"
#include "utils.h"
#include "dag_vector.cpp"
#include "dag_rectangular.cpp"

edge_t* adj = NULL; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
int* cost_on_processors = NULL; //adj è un array in modo da passarlo direttamente alla GPU senza doverlo convertire.

template<class T>
Graph<T>::Graph(int len, int processor_count) {
	max_children_for_nodes = max_parents_for_nodes = 0;
	n = m = 0;
	this->len = len;
	adj_len = len * len;
	adj_reverse_len = len * len;
	number_of_processors = processor_count;
	cost_on_processors_lenght = len * processor_count;
	cost_on_processors = DBG_NEW int[cost_on_processors_lenght]; //righe task, colonne processori
	nodes = DBG_NEW T[len];
	for (int i = 0; i < len; i++) nodes[i] = 0;
	for (int i = 0; i < cost_on_processors_lenght; i++) cost_on_processors[i] = -1;
	adj = NULL;
}

template<class T>
Graph<T>::~Graph() {
	delete[] nodes;
	delete[] cost_on_processors;
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
Graph<T>* Graph<T>::insertCostForProcessor(int indexOfNode, int indexOfProcessor, int cost) {
	int matrixToArrayIndex = matrix_to_array_indexes(indexOfNode, indexOfProcessor, number_of_processors);
	cost_on_processors[matrixToArrayIndex] = cost;
	return this;
}

template<class T>
Graph<T>* Graph<T>::insertEdge(T a, T b, int weight) {
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
int* Graph<T>::GetCostsArray(){
	return cost_on_processors;
}

template<class T>
edge_t* Graph<T>::GetWeightsArray(){
	return adj;
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
	int n_processors = 0;
	data_set >> n_nodes >> n_processors;

#if VECTOR_ADJ
	Graph<int>* DAG = DBG_NEW GraphVector<int>(n_nodes, n_processors);
#elif RECTANGULAR_ADJ
	Graph<int>* DAG = DBG_NEW GraphRectangular<int>(n_nodes, n_processors);
#else
	Graph<int>* DAG = DBG_NEW Graph<int>(n_nodes, n_processors);
#endif

	if (!DAG) {
		data_set.close();
		fprintf(stderr, "%s\n", "failed to allocate graph");
		exit(1);
	}

	//leggo tutti gli id in prima posizione in modo da creare la dag senza adj per il momento.
	DAG->max_parents_for_nodes = 0;
	DAG->max_children_for_nodes = 0;
	int* parentCountForNodes = DBG_NEW int[n_nodes];

	for (int i = 0; i < n_nodes; i++)
	{
		parentCountForNodes[i] = 0;
	}

	int value, cost_processor, n_successor, successor_index, data_transfer;
	for (int node = 0; node < n_nodes; node++) {
		data_set >> value;
		int current_node_index = DAG->insertNode(value);
		
		for (int i = 0; i < n_processors; i++)
		{
			data_set >> cost_processor;
			DAG->insertCostForProcessor(current_node_index, i, cost_processor);
		}

		data_set >> n_successor;
		DAG->max_children_for_nodes = max(DAG->max_children_for_nodes, n_successor);
		for (int i = 0; i < n_successor; i++)
		{
			data_set >> successor_index >> data_transfer;
			parentCountForNodes[successor_index]++;
		}
	}

	DAG->max_parents_for_nodes = parentCountForNodes[0];
	for (int i = 1; i < n_nodes; i++)
	{
		if (parentCountForNodes[i] > DAG->max_parents_for_nodes) {
			DAG->max_parents_for_nodes = parentCountForNodes[i];
		}
	}

	data_set.clear();
	data_set.seekg(0);
	data_set >> n_nodes >> n_processors;

	//adesso leggo di nuovo per creare la adj
	int current_node_index;
	for (current_node_index = 0; current_node_index < n_nodes; current_node_index++) {
		data_set >> value;

		for (int i = 0; i < n_processors; i++)
		{
			data_set >> cost_processor;
		}

		data_set >> n_successor;
		for (int i = 0; i < n_successor; i++)
		{
			data_set >> successor_index >> data_transfer;
			if (successor_index <= current_node_index) {
				cerr << "si sta tentando di aggiungere un arco che potrebbe causare un loop e rendere quindi il grafico ciclico" << endl;
				continue;
			}
			//questo è più veloce perché crea meno nodi in quanto nodi con lo stesso peso vengono considerati lo stesso nodo: DAG->insertEdgeByIndex(DAG->indexOfNode(value), successor_index, 1/*data_transfer*/); //assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
			DAG->insertEdgeByIndex(current_node_index, successor_index, data_transfer); //assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
		}
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