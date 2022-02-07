#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include "dag.h"
#include <random>

#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <map>

#include <math.h>

using namespace std;

static int MAX_CHILD = 10;
static int MAX_TASK_WEIGHT = 100;
static int MAX_LINK_WEIGHT = 50;
static string space = " ";

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./generator <number of task> <output file>" << endl;
        return 0;
    }

    int random_counter = 0;
    srand(time(NULL));
    //srand(1);
    string file_name = argv[2];
    int n_task = stoi(argv[1]);
    stringstream ss;
    ss << "./data_set/" << file_name << ".txt";
    string file_path = ss.str();
    
    ofstream file;
    file.open(file_path);

    if(!file.is_open()){
	    error("impossibile aprire il dataset");
	}

    file << n_task << endl;
    bool *child_occupied = new bool[n_task];
    for (int i = 0; i < n_task; i++)
    {
        // cout<< "task " << i << endl;
        //TODO: geestire frequenza dei valori, ad esempio vorrei che lo zero uscisse meno frequentemente quando si tratta di scegliere il numero di child.
        int max_child_possible = min((n_task-i-1), MAX_CHILD);
        //int n_child = (g() % max_child_possible); //dovrebbe essere un intero compreso tra l'indice del task attuale e il massimo indice possibile.
        int n_child = 0;
        if(max_child_possible > 0){
            n_child = (rand() % max_child_possible); //dovrebbe essere un intero compreso tra l'indice del task attuale e il massimo indice possibile.
            if(n_child == 0) n_child = (rand() % max_child_possible); //in questo modo rendiamo meno probabile il valore zero.
        }
        int task_weight = (rand() % MAX_TASK_WEIGHT) + 1;
        file << task_weight << space << n_child << space;
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
            file << child_index << space << link_weight << space;
            child_occupied[child_index] = 1;
        }

        file << endl;
    }

    cout<< "exit " << endl;
    file.flush();
    file.close();
    
    return 0;
}




