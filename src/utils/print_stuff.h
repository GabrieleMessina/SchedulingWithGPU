#include <iostream>
using namespace std;

template<typename T>
void print(T *v, int len, string separator = " ", bool withIndexes = false){
	for (int i = 0; i < len; i++)
	{
		if(withIndexes)
			cout << i << ":";
		cout<<v[i]<<separator;
	}
	
	cout<<"\n";
}


template<typename T>
bool isEmpty(T *v, int len, T default_v = 0){
	return howManyGreater(v, len, default_v) == 0;
}

template<typename T>
int howManyGreater(T *v, int len, T default_v = 0){
	int elements = 0;
	for (int i = 0; i < len; i++)
	{
		if(v[i] > default_v)
			elements++;
	}
	//cout<<"in coda ci sono: "<<elements<<" elementi\n";
	return elements;
}

template<typename T>
void print(T **v, int row_len, int col_len, string separator = " "){
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			cout<<v[i][j]<<separator;
		}
		cout<<"\n";
	}
	
	cout<<"\n";
}

template<typename T>
void print(T *v, int row_len, int col_len, string separator = " "){
	int mat_len = row_len * col_len;
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			cout<<v[matrix_to_array_indexes(i,j,row_len)]<<separator;
		}
		cout<<"\n";
	}
	
	cout<<"\n";
}