#include "utils.h"
#include <math.h>

void error(char const* str) {
	fprintf(stderr, "%s\n", str);
	exit(1);
}


int matrix_to_array_indexes(int i, int j, int row_len) {
	return i * row_len + j;
}

bool isEven(int a) {
	return (a % 2) == 0;
}

bool isEmpty(cl_int4* v, int len, cl_int4 default_v) {
	for (int i = 0; i < len; i++) {
		if (v[i].s0 != default_v.s0 || v[i].s1 != default_v.s1 || v[i].s2 != default_v.s2 || v[i].s3 != default_v.s3)
			return false;
	}
	return true;
}

bool isEmpty(int* v, int len, int default_v) {
	for (int i = 0; i < len; i++)
	{
		if (v[i] != default_v)
			return false;
	}
	return true;
}

bool is_file_empty(FILE* fp) {
	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);
	rewind(fp);
	return size == 0;
}

size_t GetMetricsArrayLenght(int n_nodes) {
	return pow(2, ceil(log(n_nodes) / log(2)));
}



void print(cl_int2* v, int len, const char* separator, bool withIndexes) {
	for (int i = 0; i < len; i++)
	{
		if (withIndexes)
			std::cout << i << ":";
		std::cout << "(" << v[i].s[0] << ", " << v[i].s[1] << ")" << separator;
	}
	std::cout << "\n";
}

void print(cl_int4* v, int len, const char* separator, bool withIndexes) {
	for(int i = 0; i < len; i++)
	{
		if (withIndexes)
			std::cout << i << ":";
		std::cout << "(" << v[i].s[0] << ", " << v[i].s[1] << ", " << v[i].s[2] << ", " << v[i].s[3] << ")" << separator;
	}
	std::cout << "\n";
}



/// <summary>
/// TODO: Print only if debug enabled and add to the .h file
/// </summary>
int printd(char const* const _Format, ...) {
	int _Result;
	va_list _ArgList;
	__crt_va_start(_ArgList, _Format);
	_Result = _vfprintf_l(stdout, _Format, NULL, _ArgList);
	__crt_va_end(_ArgList);
	return _Result;
}


//Una l � minore di r se hanno lo stesso livello, e l ha peso maggiore o se l ha livello minore di r.
bool operator<(const cl_int2& l, const cl_int2& r) {
	if (l.y == r.y) {
		return l.x > r.x; //peso maggiore
	}
	return l.y < r.y; //o livello pi� basso
}
bool operator>(const cl_int2& l, const cl_int2& r) {
	return (r < l);
}
bool operator<=(const cl_int2& l, const cl_int2& r) {
	return !(l > r);
}
bool operator>=(const cl_int2& l, const cl_int2& r) {
	return !(l < r);
}



std::string exec(const char* cmd) {
	char buffer[128];
	std::string result = "";
	FILE* pipe = _popen(cmd, "r");
	if (!pipe) throw std::runtime_error("popen() failed!");
	try {
		while (fgets(buffer, sizeof buffer, pipe) != NULL) {
			result += buffer;
		}
	}
	catch (...) {
		_pclose(pipe);
		throw;
	}
	_pclose(pipe);
	return result;
}