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
    i = 0
    for process in number_of_processes:
        #USING VOCABULARY SIZE 100
        command = ["python3", "MRKmeans.py", "--ncores", str(process), "--prot", "prototypes_100.txt", "--docs", "documents_100.txt"]
        result = subprocess.run(command, capture_output=True, text=True)
        result_parts = result.stdout.split(",")
        
        # Convert each part to the desired data type
        first_iteration_time = float(result_parts[0])
        time = float(result_parts[1])
        iterations = int(result_parts[2])
        avg_distortion = float(result_parts[3])
        timers.append(float(time))
        first_timers.append(float(first_iteration_time))
        
        #USING VOCABULARY SIZE 250
        command = ["python3", "MRKmeans.py", "--ncores", str(process), "--prot", "prototypes_250.txt", "--docs", "documents_250.txt"]
        result = subprocess.run(command, capture_output=True, text=True)
        result_parts = result.stdout.split(",")
        # Convert each part to the desired data type
        first_iteration_time = float(result_parts[0])
        time = float(result_parts[1])
        iterations = int(result_parts[2])
        avg_distortion = float(result_parts[3])
        timers_2.append(float(time))
        first_timers_2.append(float(first_iteration_time))
        print(f"loading {i}")
        i += 1
        
    with open("process_timers_1.json", "w") as f:
        json.dump(timers,f)
        
    with open("process_timers_2.json", "w") as f:
        json.dump(timers_2,f)
        
    with open("process_first_timers_1.json", "w") as f:
        json.dump(first_timers,f)
        
    with open("process_first_timers_2.json", "w") as f:
        json.dump(first_timers_2,f)
        
def show_plot_for_processes_exec_time():
    number_of_processes = np.arange(1,20)
    with open("process_timers_1.json", "r") as f:
        timers = json.load(f)

    with open("process_timers_2.json", "r") as f:
        timers_2 = json.load(f)
        
    plt.plot(number_of_processes, timers, color="red", label="mida 100")
    plt.plot(number_of_processes, timers_2, color="blue", label="mida 250")
    plt.legend()
    plt.xlabel('Nombre de processadors')
    plt.ylabel("Temps d'execució")
    plt.title("Temps d'execució en funció del nombre de processos ")
    plt.show()

def show_plot_for_processes_first_it_exec_time():
    
    number_of_processes = np.arange(1,20)
    with open("process_first_timers_1.json", "r") as f:
        first_timers = json.load(f)

    with open("process_first_timers_2.json", "r") as f:
        first_timers_2 = json.load(f)

    plt.plot(number_of_processes, first_timers, color="red", label="mida 100")
    plt.plot(number_of_processes, first_timers_2, color="blue", label="mida 250")
    plt.legend()
    plt.xlabel("Nombre de processadors")
    plt.ylabel("Temps d'execució")
    plt.title("Temps d'execució en funció del nombre de processos de la primera iteració")
    plt.show()
    
def show_elbow_method_plot():
    number_of_clusters = np.arange(1,20)
    with open("cluster_avg_distortion.json", "r") as f:
        avg_distortion = json.load(f)

    with open("cluster_avg_distortion.json", "r") as f:
        avg_distortion2 = json.load(f)
        
    plt.plot(number_of_clusters, avg_distortion, color="red", label="mida 100")
    plt.plot(number_of_clusters, avg_distortion2, color="blue", label="mida 250")
    plt.legend()
    plt.xlabel("Nombre de clústers")
    plt.ylabel("Distorsió mitjana")
    plt.title("Distorsió mitjana (En totes les iteracions d'una execució) en funció del nombre de clústers")
    plt.show()
    
def show_cluster_exec_time_plot():
    number_of_clusters = np.arange(1,20)
    with open("cluster_exec_time.json", "r") as f:
        exec_time = json.load(f)

    with open("cluster_exec_time_2.json", "r") as f:
        exec_time_2 = json.load(f)
        
    plt.plot(number_of_clusters, exec_time, color="red", label="mida 100")
    plt.plot(number_of_clusters, exec_time_2, color="blue", label="mida 250")
    plt.legend()
    plt.xlabel("Nombre de clústers")
    plt.ylabel("Temps d'execució")
    plt.title("Temps d'execució en funció del nombre de clústers")
    plt.show()
    
def run_execution_time_and_elbow_methods_centroid_experiments() -> None: 
    number_of_clusters = np.arange(1,20)
    exec_times = []
    average_distortions = []
    exec_times_2 = []
    average_distortions_2 = []
    
    i = 0
    for n_cluster in number_of_clusters:
        #USING VOCABULARY SIZE 100
        gen_command = ["python3", "GeneratePrototypes.py", "--nclust", str(n_cluster), "--data", "documents_100.txt"]
        result = subprocess.run(gen_command, capture_output=True, text=True)
        
        command = ["python3", "MRKmeans.py", "--prot", "prototypes.txt", "--docs", "documents_100.txt"]
        result = subprocess.run(command, capture_output=True, text=True)
        result_parts = result.stdout.split(",")
        print(result.stdout)
        # Convert each part to the desired data type
        first_iteration_time = float(result_parts[0])
        time = float(result_parts[1])
        iterations = int(result_parts[2])
        average_distortion = float(result_parts[3])
        
        exec_times.append(time)
        average_distortions.append(average_distortion)
        
        #USING VOCABULARY SIZE 250
        gen_command = ["python3", "GeneratePrototypes.py", "--nclust", str(n_cluster), "--data", "documents_250.txt"]
        result = subprocess.run(gen_command, capture_output=True, text=True)

        
        command = ["python3", "MRKmeans.py", "--prot", "prototypes.txt", "--docs", "documents_250.txt"]
        result = subprocess.run(command, capture_output=True, text=True)
        result_parts = result.stdout.split(",")
        print(result.stdout)

        # Convert each part to the desired data type
        first_iteration_time = float(result_parts[0])
        time = float(result_parts[1])
        average_distortion = float(result_parts[3])
        iterations = int(result_parts[2])
        
        exec_times_2.append(time)
        average_distortions_2.append(average_distortion)
        print(f"loading {i}")
        i += 1
        
    with open("cluster_exec_time.json", "w") as f:
        json.dump(exec_times,f)
    with open("cluster_avg_distortion.json", "w") as f:
        json.dump(average_distortions, f)
    with open("cluster_exec_time_2.json", "w") as f:
        json.dump(exec_times_2,f)
    with open("cluster_avg_distortion_2.json", "w") as f:
        json.dump(average_distortions_2,f)
        
        
        

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
    x = np.linspace(0.1, 1.0, num=8, endpoint=False)
    y = np.linspace(0.1, 1.0, num=8, endpoint=False)
    z = []
    exec_time = []
    i = 0
    for m in x:
        for M in y:
            command = ["python3", "ExtractData.py","--index","arxiv_kmeans", "--minfreq", str(m), "--maxfreq", str(M)]
            # Use subprocess.run to execute the command and capture the output
            timer1 = time.time()
            subprocess.run(command, capture_output=True, text=True)
            timer2 = time.time()

            print(f"loading {i}")  
            file = open("vocabulary.txt", "r")
            z.append(len(file.readlines()))
            file.close()
            exec_time.append(timer2-timer1)
            i+= 1
    with open("vocabulary_experiment.json", "w") as f: 
        json.dump(z, f)
    with open("vocabulary_time_experiment.json", "w") as f:
        json.dump(exec_time, f)

  
def show_vocabulary_size_experiment():
    m_values = np.linspace(0.1, 1.0, num=8, endpoint=False)
    M_values = np.linspace(0.1, 1.0, num=8, endpoint=False)
    x = []
    y = []
    
    with open("vocabulary_experiment.json", "r") as f:
        z = json.load(f)
        
    for m in m_values:
        for M in M_values:
            x.append(m)
            y.append(M)
            #z is already loaded

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
    
def show_vocabulary_time_experiment():
    m_values = np.linspace(0.1, 1.0, num=8, endpoint=False)
    M_values = np.linspace(0.1, 1.0, num=8, endpoint=False)
    x = []
    y = []
    
    with open("vocabulary_time_experiment.json", "r") as f:
        z = json.load(f)
        
    for m in m_values:
        for M in M_values:
            x.append(m)
            y.append(M)
            #z is already loaded

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.plot_trisurf(x,y,z,color=(0.85,0,0,0.8))

    ax.set_xlabel('m')
    ax.set_ylabel('M')
    ax.set_zlabel("Temps d'execució (Segons)")
    ax.set_title("Temps d'execució en funció de les frequencies Máximes i Mínimes de filtratge")
    plt.show()

    
def show_cluster_most_important_words(output_file):
    most_imp = (get_most_frequent_word_foreach_prototype("./prototypes_final.txt", 10))
    
    with open(output_file, "w") as f:
        json.dump(most_imp, f)
    
def execute_k_means_and_show_most_important_words():    
    command = ["python3", "MRKmeans.py", "--prot", "prototypes_100.txt", "--docs", "documents_100.txt"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    show_cluster_most_important_words("10_prototypes_100.json")
    
    command = ["python3", "MRKmeans.py", "--prot", "prototypes_250.txt", "--docs", "documents_250.txt"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    show_cluster_most_important_words("10_prototypes_250.json")
    
    gen_command = ["python3", "GeneratePrototypes.py", "--nclust", "6", "--data", "documents_100.txt"]
    result = subprocess.run(gen_command, capture_output=True, text=True)
    
    command = ["python3", "MRKmeans.py", "--prot", "prototypes.txt", "--docs", "documents_100.txt"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    show_cluster_most_important_words("6_prototypes_100.json")
    
    gen_command = ["python3", "GeneratePrototypes.py", "--nclust", "6", "--data", "documents_250.txt"]
    result = subprocess.run(gen_command, capture_output=True, text=True)
    
    command = ["python3", "MRKmeans.py", "--prot", "prototypes.txt", "--docs", "documents_250.txt"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    show_cluster_most_important_words("6_prototypes_250.json")
    
def show_pretty_prototypes_qualitative_experiment(): 
    with open("6_prototypes_100.json", "r") as f:
        protos_6_100 = json.load(f)
        protos_6_100 = json.dumps(f, indent=4)
        print(protos_6_100)
    with open("6_prototypes_250.json", "r") as f:
        protos_6_250 = json.load(f)
        protos_6_250 = json.dumps(f, indent=4)
        print(protos_6_250)
    with open("10_prototypes_100.json", "r") as f:
        protos_10_100 = json.load(f)
        protos_10_100 = json.dumps(f, indent=4)
        print(protos_10_100)
    with open("10_prototypes_250.json", "r") as f:
        protos_10_250 = json.load(f)
        protos_10_250 = json.dumps(f, indent=4)
        print(protos_10_250)
        
        



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
    
    
    #run_experiment_of_execution_time_and_number_of_processes()
    #run_experiment_of_execution_time_and_number_of_processes()
    #run_execution_time_and_elbow_methods_centroid_experiments()
    show_plot_for_processes_exec_time()
    
    #show_plot_for_processes_first_it_exec_time()
    
    #show_elbow_method_plot()
    
    #show_cluster_exec_time_plot()
    
    #show_vocabulary_size_experiment()
    
    #show_vocabulary_time_experiment()
    
    #show_pretty_prototypes_qualitative_experiment()
    
    
