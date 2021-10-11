#include<iostream>
using namespace std;

/*IMPLEMENTAZIONE DEL NODO DELLA DAG CHE MANTIENE I DATI NECESSARI ALL'ALGORITMO*/
class GraphNode{
	public:
		int id;
		int value;
		GraphNode(int value){
			this->id = 0; //TODO: verificare quali sono i dati effettivi da mantenere.
			this->value = value;
		}
		friend ostream& operator<<(ostream& os, const GraphNode& dt);
};

ostream& operator<<(ostream& os, const GraphNode& dt)
{
    os << "id: " << dt.id << ", value:" << dt.value;
    return os;
}

bool operator == (const GraphNode& lhs, const GraphNode& rhs)
{
    return lhs.value == rhs.value;
}