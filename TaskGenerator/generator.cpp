#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <random>

#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <map>

#include <math.h>

using namespace std;

static int MAX_TASK_WEIGHT = 100;
static int MAX_LINK_WEIGHT = 50;
static string space = " ";

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}
string file_name;
int n_task;
int main(int argc, char const *argv[])
{
    if (argc <= 1) {
        cout << "Number of tasks: ";
        cin >> n_task;
        cout << "Output file name (no path, no extensions): ";
        cin >> file_name;
    }
    else if(argc != 3)
    {
        cout << "Usage: ./generator <number of tasks> <output file name>" << endl;
        return 0;
    }
    else {
        file_name = argv[2];
        n_task = stoi(argv[1]);
    }

    int random_counter = 0;
    srand(time(NULL));
    //srand(1);
     
    stringstream ss;
    ss << "./data_set/" << file_name << ".txt";
    string file_path = ss.str();
    
    FILE* fp;
    fp = fopen(file_path.c_str(), "w+");

    if(fp == NULL){
	    error("impossibile aprire il dataset");
	}

    fprintf(fp, "%d\n", n_task);

	static int MAX_CHILD = 10;
	MAX_CHILD = (n_task / 100) + 10; //1% of n_task, with +10 for smallest data set
    bool *child_occupied = new bool[n_task];
    for (int i = 0; i < n_task; i++)
    {
        //TODO: geestire frequenza dei valori, ad esempio vorrei che lo zero uscisse meno frequentemente quando si tratta di scegliere il numero di child.
        int max_child_possible = min((n_task-i-1), MAX_CHILD);
        //int n_child = (g() % max_child_possible); //dovrebbe essere un intero compreso tra l'indice del task attuale e il massimo indice possibile.
        int n_child = 0;
        if(max_child_possible > 0){
            n_child = (rand() % max_child_possible); //dovrebbe essere un intero compreso tra l'indice del task attuale e il massimo indice possibile.
            if(n_child == 0) n_child = (rand() % max_child_possible); //in questo modo rendiamo meno probabile il valore zero.
        }
        int task_weight = (rand() % MAX_TASK_WEIGHT) + 1;
        fprintf(fp, "%d %d ", task_weight, n_child);
        for (int j = i; j < n_task; j++){ child_occupied[j] = (j<=i) ? 1 : 0;} //inizializzo i posti occupati mettendo quelli precedenti al parent e il parent stesso come già occupati in modo che l'algoritmo non li scelga.
        
        int child_index;
        int link_weight;
        for (int j = 0; j < n_child; j++)
        {
            do
            {
                //srand(time(NULL));
                srand(i+j+random_counter++);
                child_index = i + (rand() % max_child_possible); //mi assicura che l'indice del child sia maggiore di quello del padre ma comunque minore del numero di task
            } 
            while (child_index <= i || child_occupied[child_index] == 1);
            //while(child_occupied[child_index] == 1);
            //while (child_index <= i);
            
            link_weight = (rand() % MAX_TASK_WEIGHT) + 1;
            fprintf(fp,"%d %d ", child_index, link_weight);
            child_occupied[child_index] = 1;
        }

        fprintf(fp, "\n");

		/*Progress bar*/
		float progress = (i+1)/(float)n_task;
		int barWidth = 70;

		std::cout << "[";
		int pos = barWidth * progress;
		for (int cell = 0; cell < barWidth; ++cell) {
			if (cell < pos) std::cout << "=";
			else if (cell == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << i+1 << " / " << n_task << "\r";
		std::cout.flush();
    }

    std::cout.flush();
    std::cout << std::endl;
    cout<< "FINISH" << endl;
    std::cout.flush();
    fflush(fp);
    fclose(fp);
    
    return 0;
}




