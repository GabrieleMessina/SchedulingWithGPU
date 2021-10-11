/* IMPLEMENTAZIONE DELLA DAG (DIRECT ACYCLIC GRAPH)*/

#include<iostream>
using namespace std;

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
	bool **adj;
	int *color, *p, *d, *f; //colore node, predecessore, distanza o inizio visita, fine visisita
	
	Graph(int len = 100){
		n = m = 0;
		this->len = len;
		nodes = new T[len];
        for(int i = 0; i<len; i++) nodes[i] = 0;
		adj = new bool*[len];
		for(int i = 0; i<len; i++){
			adj[i] = new bool[len];
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
	
	Graph<T> *insertNode(T key){
		if(n < len){
			nodes[n++] = key; 
		}
		return this;
	}
	
	int indexOfNode(T key){
		for(int i =0; i<n; i++) if(nodes[i] == key) return i;
		return -1;
	}
	
	Graph<T> *insertEdge(T a, T b){
		int i = indexOfNode(a);
		int j = indexOfNode(b);
		if(i != -1 && j != -1){
			adj[i][j] = 1;
			//adj[j][i] = 1; //se non orientato
			m++;
		}
		return this;
	}
	
	bool hasEdge(T a, T b){
		int i = indexOfNode(a);
		int j = indexOfNode(b);
		if(i != -1 && j != -1){
			return adj[i][j];
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
				if(adj[x][i] && color[i]==0){
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
			for(int i=0; i<n; i++) {
				cout << "(" << i << ", " << nodes[i] << ")" << " : ";
				for(int j=0; j<n; j++) {
					if(adj[i][j]) cout << nodes[j] << " ";
				} 
				cout << endl;
			}
		}


    /*USAGE:
    
    Graph<GraphNode> *adj = new Graph<GraphNode>(nels);
	GraphNode a = GraphNode(321);
	GraphNode b = GraphNode(654);

	adj->insertNode(a);
	adj->insertNode(b);
	adj->insertEdge(adj->indexOfNode(a), adj->indexOfNode(b));
	adj->Print();

    */
	
	//topologicalSort, BFS, distance, componentiFortemente, tempidivisita
};


// int main(){
	
// 	Graph<int> *g = new Graph<int>(10);
// 	g->insertNode(0)->insertNode(1)->insertNode(2)->insertNode(3)->insertNode(4)->insertNode(5)->insertNode(6)->insertNode(7)->insertNode(8);
	
// 	g->insertEdge(0,8)->insertEdge(0,1);
// 	g->insertEdge(1,8);
// 	g->insertEdge(2,4);
// 	g->insertEdge(3,5)->insertEdge(3,6)->insertEdge(3,7);
// 	g->insertEdge(4,3)->insertEdge(4,0);
// 	g->insertEdge(5,6)->insertEdge(5,3);
// 	g->insertEdge(6,5);
// 	g->insertEdge(8,2);
	
// 	/*for(int i=0; i<g->len; i++){
// 		for(int j=0; j<g->len; j++){
// 			cout<<g->adj[i][j]<< " ";
// 		}	
// 		cout<<endl;
// 		cout<<*g->v[i]<< " ";
// 	}cout<<endl;*/
	
// 	//g->Print();
// 	g->DFS();
	
	
// 	return 0;
// }



/*IMPLEMENTAZIONE WIP CON VECTOR INVECE CHE ARRAY E MATRICI*/
// class DAG{ //Direct Acyclic Graph
// 	private: 
// 		int startSize = 20;
// 	public:
// 		vector<GraphNode> nodes; // o GraphNode *nodes;
// 		int nodesNumber = 0;
// 		int **adj; // o vector<vector<int>> adj;
// 		int edgeNumber = 0; //inutile se si usa la matrice di adiacenza invece della lista


// 		DAG(int nodesNumber = 100){ //TODO: il nodesNumber non è detto che sia conosciuto a priori, eventualemente gestire la adj con vector, che fra l'altro è conveniente anche algoritmicamente.
// 			nodes = vector<GraphNode>(startSize); // o new GraphNode[startSize]
// 			//adj = vector<int>(startSize); // o new int[startSize]; for( i : startSize) new int[startSize][i];
// 			this->nodesNumber = nodesNumber;
// 			adj = new int*[nodesNumber];
// 			for (int i = 0; i < nodesNumber; i++){
// 				adj[i] = new int[nodesNumber];
// 				for (int j = 0; j < nodesNumber; j++)
// 					adj[i][j] = 0;
// 			}
// 		}

// 		DAG InsertNode(GraphNode node){
// 			nodes.push_back(node);
// 			nodesNumber++;
// 			return this;
// 		}

// 		void InsertEdge(GraphNode node){
// 			adj.push_back(node);
// 			edgeNumber++;
// 		}


// };
