#include <iostream>
using namespace std;

void print(int *v, int len){
	for (int i = 0; i < len; i++)
	{
		cout<<v[i]<<" ";
	}
	
	cout<<"\n";
}

void print(int **v, int row_len, int col_len){
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			cout<<v[i][j]<<" ";
		}
		cout<<"\n";
	}
	
	cout<<"\n";
}