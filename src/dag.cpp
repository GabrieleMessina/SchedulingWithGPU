#include "dag.h"
#include "utils.h"

template<class T>
Graph<T>::Graph(int len) {
	n = m = 0;
	this->len = len;
	adj_len = len * len;
	nodes = DBG_NEW T[len];
	for (int i = 0; i < len; i++) nodes[i] = 0;

#if VECTOR_ADJ
	adj.reserve(adj_len);
	adj.resize(adj_len);
#else
	adj = DBG_NEW edge_t[adj_len];
	#pragma unroll
	for (int i = 0; i < adj_len; i++) {
		adj[i] = 0;
	}
#endif

	printf("len is %d and mat is %d\n", len, adj_len);
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
Graph<T>* Graph<T>::insertEdge(T a, T b, int weight) {
	//TODO: verifica che non crei cicli!
	int i = indexOfNode(a);
	int j = indexOfNode(b);
	
	return insertEdgeByIndex(a, b, weight);
}

template<>
edge_t* Graph<int>::GetEdgesArray(){
#if VECTOR_ADJ
	return adj.data();
#else
	return adj;
#endif
}

template<class T>
Graph<T>* Graph<T>::insertEdgeByIndex(int indexOfa, int indexOfb, int weight) {
	int i = indexOfa;
	int j = indexOfb;
	if (i > -1 && j > -1 && i < len && j < len) {
#if VECTOR_ADJ
		adj.insert(adj.begin() + matrix_to_array_indexes(i, j, len), weight);
#else
		adj[matrix_to_array_indexes(i, j, len)] = weight;
#endif
		m++;
	}
	else {
		if (typeid(int) == typeid(indexOfa)) {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
		else if (typeid(string) == typeid(indexOfa)) {
			printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
		}
	}
	return this;
}

template<class T>
bool Graph<T>::hasEdge(T a, T b) {
	int i = indexOfNode(a);
	int j = indexOfNode(b);
	if (i != -1 && j != -1) {
#if VECTOR_ADJ
		return adj.at(matrix_to_array_indexes(i, j, len)) != 0;
#else
		return adj[matrix_to_array_indexes(i, j, len)] != 0;
#endif
		
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

	Graph<int>* DAG = DBG_NEW Graph<int>(n_nodes);
	if (!DAG) {
		data_set.close();
		fprintf(stderr, "%s\n", "failed to allocate graph");
		exit(1);
	}

	//leggo tutti gli id in prima posizione in modo da creare la dag senza adj per il momento.
	int value, n_successor, successor_index, data_transfer;
	while (data_set >> value >> n_successor) {
		int current_node_index = DAG->insertNode(value);
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