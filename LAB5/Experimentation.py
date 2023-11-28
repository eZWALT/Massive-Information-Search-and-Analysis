import subprocess 
import json
import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

def run_experiment_of_execution_time_and_number_of_processes() -> None:
    
    is_flag_on = True
       #iterations experiment
    number_of_processes = np.arange(1,20)
    timers = []
    timers_2 = []
    first_timers = []
    first_timers_2 = []
    for process in number_of_processes:
        #USING VOCABULARY SIZE 100
        command = ["python3", "MRKmeans.py", "--ncores", str(process), "--prot", "prototype_100.txt", "--docs", "documents_100.txt"]
        result = subprocess.run(command, capture_output=True, text=True)
        first_iteration_time, time, iterations, convergence = result.stdout.split(",")
        timers.append(time)
        first_timers.append(first_iteration_time)
        
        #USING VOCABULARY SIZE 250
        command = ["python3", "MRKmeans.py", "--ncores", str(process), "--prot", "prototype_250.txt", "--docs", "documents_250.txt"]
        result = subprocess.run(command, capture_output=True, text=True)
        first_iteration_time, time, iterations, convergence = result.stdout.split(",")
        timers_2.append(time)
        first_timers_2.append(first_iteration_time)
        
        plt.plot(number_of_processes, timers, color="red", label="mida 100")
        plt.plot(number_of_processes, timers_2, color="blue", label="mida 250")
        plt.legend()
        plt.xlabel('Nombre de processadors')
        plt.ylabel("Temps d'execució")
        plt.title("Temps d'execució en funció del nombre de processos ")
        plt.show()
        
                
        plt.plot(number_of_processes, first_timers, color="red", label="mida 100")
        plt.plot(number_of_processes, first_timers_2, color="blue", label="mida 250")
        plt.legend()
        plt.xlabel("Nombre de processadors")
        plt.ylabel("Temps d'execució")
        plt.title("Temps d'execució en funció del nombre de processos de la primera iteració")
        plt.show()
        
        
def run_execution_time_and_elbow_methods_centroid_experiments() -> None: 
    pass
        

def get_most_frequent_word_foreach_prototype(proto_file: str, k: int) -> {int: str}:
    file = open(proto_file, "r")
    i = 0
    clusters = {}
    for line in file.readlines():
        proto, wordlist = line.split(":")
        prototype = {}
        for metaword in wordlist.split():
            word, freq = metaword.split("+")
            prototype[word] = freq 
        
        sorted_prototype = dict(sorted(prototype.items(),
                                  key=lambda item: item[1],
                                  reverse=True
                                  )
                           )
        get_k_biggest = dict(list(sorted_prototype.items())[:k])
        clusters[i] = get_k_biggest
        i += 1
    file.close()
    return clusters

#First two experiments
def run_experiment_of_size_of_vocab_as_function_of_m_and_M_and_show_plot() -> None:
    x = np.linspace(0.1, 1.0, num=10, endpoint=True)
    y = np.linspace(0.1, 1.0, num=10, endpoint=True)
    z = []
    exec_time = []
    for m in x:
        for M in y:
            command = ["python3", "ExtractData.py","--index","arxiv_kmeans", "--minfreq", str(m), "--maxfreq", str(M)]
            # Use subprocess.run to execute the command and capture the output
            timer1 = time.time()
            subprocess.run(command, capture_output=True, text=True)
            timer2 = time.time()

            file = open("vocabulary.txt", "r")
            z.append(len(file.readlines()))
            file.close()
            exec_time.append(timer2-timer1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.plot_trisurf(x,y,z,color=(0.85,0,0,0.8))

    ax.set_xlabel('m')
    ax.set_ylabel('M')
    ax.set_zlabel('Mida del Vocabulari')
    ax.set_title('Mida del Vocabulari en funció de les frequencies Máximes i Mínimes de filtratge')
    plt.show()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.plot_trisurf(x,y,exec_time,color=(0.85,0,0,0.8))

    ax.set_xlabel('m')
    ax.set_ylabel('M')
    ax.set_zlabel("Temps d'execució (Segons)")
    ax.set_title("Temps d'execució en funció de les frequencies Máximes i Mínimes de filtratge")
    plt.show()


def show_cluster_most_important_words():    
    most_imp = json.dumps(get_most_frequent_word_foreach_prototype("./prototypes_final.txt", 10), indent=4)    
    print(most_imp)


if __name__ == "__main__":
    
    #default input parameters
    max_freq = 0.1
    min_freq = 0.05 
    num_iterations = 100
    num_processes = 8
    
    #output values 
    # num_iterations
    # execution_time
    # The clusters itself 
        