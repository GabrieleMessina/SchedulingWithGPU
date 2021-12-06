/* IMPLEMENTAZIONE DELLA DAG (DIRECT ACYCLIC GRAPH)*/
using namespace std;

int matrix_to_array_indexes(int i, int j, int row_len){
	return i * row_len + j;
}


template<typename T> T* matrix_to_array(T **mat, int row_len, int col_len){
	T *v = new T[row_len * col_len];
	int t = 0;
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			v[t++] = mat[i][j];
		}	
	}
	return (T*) &v[0];
}

template<typename T> void matrix_to_array(T **mat, int row_len, int col_len, T* outArray){
	int t = 0;
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			outArray[t++] = mat[i][j];
		}	
	}
}


template<class T> class Queue{
	public:
	int head, tail, len;
	T *nodes;
	
	Queue(int len = 100){
		this->len = len;
		head = tail = -1;
		nodes = new T[len];
		for(int i=0; i<len; i++) nodes[i] = -1;
	}
	
	void Enqueue(T key){
		if(tail < len && head<=tail){
			nodes[++tail] = key;
		}
	}
	
	T Dequeue(){
		if(head < len && head<=tail){
			return nodes[++head];
		}else return -1;
	}
	
	bool isEmpty(){
		return tail == head;
	}

};

/* IMPLEMENTAZIONE DELLA DAG (DIRECT ACYCLIC GRAPH)*/
template<class T> class Graph{
	public:
	int len, adj_len, n, m; //lunghezza, numero di nodi, numero di archi
	T *nodes;
	bool *adj; //TODO: adj potrebbe già essere un array in modo da passarlo direttamente alla GPU senza doverlo convertire.
	//int *color, *p, *d, *f; //colore node, predecessore, distanza o inizio visita, fine visisita
	
	~Graph(){
		free(nodes);
		free(adj);
		// free(f);
		// free(d);
		// free(p);
	}

	Graph(int len = 100){
		n = m = 0;
		this->len = len;
		adj_len = len * len;
		nodes = new T[len];
        for(int i = 0; i<len; i++) nodes[i] = 0;
		adj = new bool[adj_len];
		printf("len is %d and mat is %d\n", len , adj_len);
#pragma unroll
		for(int i = 0; i<adj_len; i++){
			adj[i] = 0;
		}

		// color = new int[len];
		// p = new int[len];
		// d = new int[len];
		// f = new int[len];
		// for(int i=0; i<len; i++){
		// 	color[i]=0;
		// 	d[i] = f[i] = len+1;
		// 	p[i] = -1;
		// }
	}
	
	int insertNode(T key){
		if(n < len){
			nodes[n] = key; 
		}
		return n++;
	}

	int insertUniqueValueNode(T key){
		if(indexOfNode(key) != -1) return -1;
		if(n < len){
			nodes[n] = key; 
		}
		return n++;
	}
	
	int indexOfNode(T key){
		for(int i =0; i<len; i++) if(nodes[i] == key) return i;
		return -1;
	}
	
	Graph<T> *insertEdge(T a, T b, int weight = 1){
		//TODO: verifica che non crei cicli!
		int i = indexOfNode(a);
		int j = indexOfNode(b);
		if(i != -1 && j != -1){
			adj[matrix_to_array_indexes(i,j,len)] = weight;
			//adj[j][i] = weight; //se non orientato
			m++;
		}
		else{
			if (typeid(int) == typeid(a)) {
				printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", a, b);
			}
			else if (typeid(string) == typeid(a)) {
				printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%s,%s)\n", a, b);
			}
		}
		return this;
	}

	Graph<T> *insertEdgeByIndex(int indexOfa, int indexOfb, int weight = 1){
		int i = indexOfa;
		int j = indexOfb;
		if(i > -1 && j > -1 && i < len && j < len){
			adj[matrix_to_array_indexes(i,j,len)] = weight;
			//adj[j][i] = weight; //se non orientato
			m++;
		}
		else{
			if (typeid(int) == typeid(indexOfa)) {
				printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%d,%d)\n", indexOfa, indexOfb);
			}
			else if (typeid(string) == typeid(indexOfa)) {
				printf("impossibile aggiungere l'edge perche' uno degli indici non esiste in insertEdge(%s,%s)\n", indexOfa, indexOfb);
			}
		}
		return this;
	}
	
	bool hasEdge(T a, T b){
		int i = indexOfNode(a);
		int j = indexOfNode(b);
		if(i != -1 && j != -1){
			return adj[matrix_to_array_indexes(i,j,len)] != 0;
		}
		return false;
	}
	
	void VisitDFS(int s){
		Queue<int> *q = new Queue<int>();
		q->Enqueue(s);
		color[s] = 1;
		p[s] = -1;
		d[s] = 0;
		while(!q->isEmpty()){
			int x = q->Dequeue();
			for(int i=0; i<n; i++){
				if(adj[matrix_to_array_indexes(x,i,len)] != 0 && color[i]==0){
					color[i] = 1;
					q->Enqueue(i);
					p[i] = x;
					d[i] = d[x]+1;
				} //else hasCycle
			}
			color[x] = 2;
		}
	}
	
	void DFS(){
		//int *color, *p, *d, *f; //colore nodo, predecessore, distanza o inizio visita, fine visisita
		for(int i=0; i<len; i++){
			color[i]=0;
			d[i] = f[i] = len+1;
			p[i] = -1;
		}
		for(int i=0; i<n; i++){
			if(color[i] == 0) VisitDFS(i);//or hasCycle;
		}
		for(int i=0; i<n; i++) {
				cout << "[" << i << "]->";
				if(d[i]==len+1) cout << "inf." << endl;
				else cout << d[i] << endl;
			}
		cout << endl;
	}
	
	void Print() {
		cout << "(index, value) -> [(edges_index, weight)]"<<endl;
		for(int i=0; i<len; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
			for(int j=0; j<len; j++) {
				if(adj[matrix_to_array_indexes(i,j,len)] != 0) cout << "(" << j << ", " << adj[matrix_to_array_indexes(i,j,len)] << ") | ";
			} 
			cout << endl;
		}
	}
};


Graph<int>* initDagWithDataSet(string dataset_file_name){
	ifstream data_set;
	stringstream ss;
	ss << "./data_set/" << dataset_file_name << ".txt";
	string dataset_file_name_with_extension = ss.str();
	data_set.open(dataset_file_name_with_extension);
	cout<<"dataset_file_name_with_extension: "<<dataset_file_name_with_extension<<endl;
	if(!data_set.is_open()){
		fprintf(stderr, "%s\n", "impossibile aprire il dataset");
		exit(1);
	}
	
	int n_nodes = 0;
	data_set >> n_nodes;
		
	Graph<int> *DAG = new Graph<int>(n_nodes);
	if (!DAG) {
		data_set.close();
		fprintf(stderr, "%s\n", "failed to allocate graph");
		exit(1);
	}

	//leggo tutti gli id in prima posizione in modo da creare la dag senza adj per il momento.
	int value, n_successor, successor_index, data_transfer;
	while(data_set >> value >> n_successor){
		int index = DAG->insertNode(value);
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor_index>>data_transfer;
		}
	}

	data_set.clear();
	data_set.seekg(0);
	data_set >> n_nodes;

	//adesso leggo di nuovo per creare la adj

	while(data_set >> value >> n_successor){
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor_index>>data_transfer;
			DAG->insertEdgeByIndex(DAG->indexOfNode(value), successor_index, 1/*data_transfer*/); //TODO: al momento assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
		}
	}
	//TODO: e se mantenessi in memoria una matrice quadrata con tutti questi dati in fila per ogni task? IN modo da avere coalescenza?

	data_set.close();
	return DAG;	
}