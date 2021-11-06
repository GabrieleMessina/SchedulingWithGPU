/* IMPLEMENTAZIONE DELLA DAG (DIRECT ACYCLIC GRAPH)*/
using namespace std;

int* matrix_to_array(int **mat, int row_len, int col_len){
	int *v = new int[row_len * col_len];
	int t = 0;
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			v[t++] = mat[i][j];
		}	
	}
	return v;
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
	int len, n, m;
	T *nodes;
	int **adj;
	int *color, *p, *d, *f; //colore node, predecessore, distanza o inizio visita, fine visisita
	
	~Graph(){
		free(nodes);
		free(adj);
		free(f);
		free(d);
		free(p);
	}

	Graph(int len = 100){
		n = m = 0;
		this->len = len;
		nodes = new T[len];
        for(int i = 0; i<len; i++) nodes[i] = 0;
		adj = new int*[len];
		for(int i = 0; i<len; i++){
			adj[i] = new int[len];
			for(int j = 0; j<len; j++){
				adj[i][j] = 0;
			}
		}
		color = new int[len];
		p = new int[len];
		d = new int[len];
		f = new int[len];
		for(int i=0; i<len; i++){
			color[i]=0;
			d[i] = f[i] = len+1;
			p[i] = -1;
		}
	}
	
	int insertNode(T key){
		if(n < len){
			nodes[n] = key; 
		}
		return n++;
	}
	
	int indexOfNode(T key){
		for(int i =0; i<n; i++) if(nodes[i] == key) return i;
		return -1;
	}
	
	Graph<T> *insertEdge(T a, T b, int weight = 1){
		//TODO: verifica che non crei cicli!
		int i = indexOfNode(a);
		int j = indexOfNode(b);
		if(i != -1 && j != -1){
			adj[i][j] = weight;
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
			adj[i][j] = weight;
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
			return adj[i][j] != 0;
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
				if(adj[x][i] != 0 && color[i]==0){
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
		for(int i=0; i<n; i++) {
			cout << "(" << i << ", " << nodes[i] << ")" << " -> | ";
			for(int j=0; j<n; j++) {
				if(adj[i][j] != 0) cout << "(" << j << ", " << adj[i][j] << ") | ";
			} 
			cout << endl;
		}
	}
};