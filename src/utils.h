#pragma once
#include "app_globals.h"
#include "CL/cl.h"
#include <iostream>

using namespace std;

void error(char const* str);
void swap(cl_mem* a, cl_mem* b);
void swap(cl_mem a, cl_mem b);
size_t GetMetricsArrayLenght(int n_nodes);

int matrix_to_array_indexes(int i, int j, int row_len);

int calc_nwg(int nels, int lws);


bool isEven(int a);
bool isEmpty(int* v, int len, int default_v = 0);
bool isEmpty(const cl_int4* v, int len, cl_int4 default_v = cl_int4{ 0,0,0,0 });
bool isEmpty(cl_int8* v, int len, cl_int8 default_v = cl_int8{ 0,0,0,0,0,0,0,0 });
bool is_file_empty(FILE* fp);

//Specialization of print template function for this types that are not standard. //TODO: to make it cleaner, override cout << operand for this types
//void print(cl_int2* v, int len, const char* separator = " ", bool withIndexes = false, int startingFrom = 0);
void print(cl_int3* v, int len, const char* separator = " ", bool withIndexes = false, int startingFrom = 0);
void print(cl_int4* v, int len, const char* separator = " ", bool withIndexes = false);

//utils for cl types
bool operator<(const metrics_t& l, const metrics_t& r);
bool operator>(const metrics_t& l, const metrics_t& r);
bool operator<=(const metrics_t& l, const metrics_t& r);
bool operator>=(const metrics_t& l, const metrics_t& r);

std::string exec(const char* cmd);
void printMemoryUsage();

//Template function cannot be implemented in .cpp file.
template<class T>
void print(T* v, int len, const char* separator, bool withIndexes) {
	for (int i = 0; i < len; i++)
	{
		if (withIndexes)
			std::cout << i << ":";
		std::cout << v[i] << separator;
	}
	std::cout << "\n";
}

template<typename T>
void print(T** v, int row_len, int col_len, const char *separator) {
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			std::cout << v[i][j] << separator;
		}
		std::cout << "\n";
	}

	std::cout << "\n";
}

template<typename T>
void print(T* v, int row_len, int col_len, const char *separator) {
	int mat_len = row_len * col_len;
	for (int i = 0; i < row_len; i++)
	{
		for (int j = 0; j < col_len; j++)
		{
			std::cout << v[matrix_to_array_indexes(i, j, row_len)] << separator;
		}
		std::cout << "\n";
	}

	std::cout << "\n";
}

template<typename T>
void maxOfSequence(T* v, int len) {
	T max = v[0];
	for (int i = 1; i < len; i++)
	{
		if (v[i] > max) {
			max = v[i];
		}
	}
	return max;
}



//template<typename T> T* matrix_to_array(T** mat, int row_len, int col_len) {
//	T* v = DBG_NEW T[row_len * col_len];
//	int t = 0;
//	for (int i = 0; i < row_len; i++)
//	{
//		for (int j = 0; j < col_len; j++)
//		{
//			v[t++] = mat[i][j];
//		}
//	}
//	return (T*)&v[0];
//}

//
//template<typename T> void matrix_to_array(T** mat, int row_len, int col_len, T* outArray) {
//	int t = 0;
//	for (int i = 0; i < row_len; i++)
//	{
//		for (int j = 0; j < col_len; j++)
//		{
//			outArray[t++] = mat[i][j];
//		}
//	}
//}

//template<typename T>
//int howManyGreater(T* v, int len, T default_v) {
//	int elements = 0;
//	for (int i = 0; i < len; i++)
//	{
//		if (v[i] > default_v)
//			elements++;
//	}
//	//std::cout<<"in coda ci sono: "<<elements<<" elementi\n";
//	return elements;
//}