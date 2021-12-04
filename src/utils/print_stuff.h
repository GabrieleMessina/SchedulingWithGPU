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
bool isEmpty(T *v, int len, T default = 0){
	for (int i = 0; i < len; i++)
	{
		if(v[i] > default)
			return false;
	}
	return true;
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