import argparse

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
from elasticsearch.client import CatClient
import json
import heapq

from TFIDFViewer import * 
import numpy as np

#Global & Special vars
__author__ = "Walter Troiani"
PRINT_INFO = True
USE_R = True

########################################


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


    return [(term,tfidf/norm) for (term,tfidf) in tw]

def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    
    summ = 0
    for term, tfidf in tw.items(): 
        summ += tfidf**2 
    norm = np.sqrt(summ)
        
    return {term: weight/norm for term, weight in tw.items()}

def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    #file_tv dictionary of (term, howmanyaparitions in doc) , (term, howmanydocumentscontain)
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    document_count = doc_count(client, index)

    tfidfw = {}
    #Where t is term, w is count, df is document-frequency
    for (t, w),(_, df) in zip(file_tv, file_df):
        
        term_frequency = w / max_freq 
        inverse_document_frequency = np.log2(document_count / df)
        #append weigth[i,d] of term i and document d 
        tfidfw[t] = term_frequency * inverse_document_frequency

    return normalize(tfidfw)

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id

##############################################################


#remove fuzzy operators
def query_to_dictionary(query):
    result = {}
    for q in query: 
        if not "^" in q:
            if "~" in q: 
                word,value = q.split("~")
            else:
                word = q
            value = 1 
        else:
            word,value = q.split("^")
            value = float(value)
        
        result[word] = value
    return result

def dictionary_to_query(dicc):
    result = []
    for word,value in dicc.items():
        result.append(f"{word}^{value}")  
    return result   
   

def compute_tfidf_average(query, related_docs, client, index,R,k):
    result = {}
    
    #initial query wtf should i do with this?????
    for word, value in query.items():
        result[word] = 0
    
    for document in related_docs:
        identifier = search_file_by_path(client,index,document.path)
        tfidf = toTFIDF(client,index,identifier)
        
        if USE_R: 
            data_list = [(-tfidf, term) for term, tfidf in tfidf.items()]
            heapq.heapify(data_list)
            tfidf = {}
            for i in range(min(R, len(data_list))):
                value, key = heapq.heappop(data_list)
                tfidf[key] = -value
        
        #merge result and document
        for word, value in tfidf.items():
            if word in result:
                result[word] += tfidf[word]
            else: 
                result[word] = value
        print(result)
            
    print({term: weight/k for term,weight in result.items()})
        
        
def get_k_most_relevant_documents_from_rocchio_rule(search,client,index,alfa,beta,nrounds,k,R,initial_query, initial_related_documents):
    refined_query = initial_query
    relevant_documents = initial_related_documents
    
    for i in range(nrounds):
        
        #firstly compute the tfidf summation
        tfidf_avg = compute_tfidf_average(refined_query,relevant_documents, client,index,R,k)
        
        #merge query & tfidf dictionary 
        beta_expansion = {term: weigth * beta for term,weigth in tfidf_avg.items()}
        alfa_query = {term: weigth * alfa for term,weigth in refined_query.items()}
        
        #How this intersection could be implemented?   
        refined_query = alfa_query + beta_expansion
        
        ref_query = dictionary_to_query(refined_query)

        print(f"round {i}, query {ref_query}")
        
        #get the k most relevant documents
        q = Q('query_string',query=ref_query[0])
        for i in range(1, len(ref_query)):
                q &= Q('query_string',query=ref_query[i])
                
        search = search.query(q)
        relevant_documents = search[0:k].execute()
        if PRINT_INFO:
            show_most_relevant_documents(relevant_documents)
        
    return refined_query, relevant_documents

def show_most_relevant_documents(relevant_documents):
    print("\n")
    for r in relevant_documents:
        print(f"SCORE={r.meta.score}")
        print(f"PATH= {r.path}")
        
    

if __name__ == "__main__":
    
    ###
    ###  1. Parse input arguments
    ### 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=None, help="Index to search")
    parser.add_argument("--k", default=5, type=int, help="Number of k most relevant documents to show")
    parser.add_argument("--R", default=3, type=int, help="Number of R most relevant terms for pruning documents")
    parser.add_argument("--nrounds", default=3, type=int, help="Number of iterations of Pseudorelevance Feedback algorithm")
    parser.add_argument("--alfa", default=1, type=float, help="Constant factor for users query in Rocchio's Rule")
    parser.add_argument("--beta", default=0.5, type=float, help="Constant factor for relevant documents in Rocchio's Rule") 
    parser.add_argument("--query", default=None, nargs=argparse.REMAINDER,help="List of words to search")

    
    args = parser.parse_args()
    index = args.index
    query = args.query
    R = args.R 
    k = args.k
    nrounds = args.nrounds 
    alfa = args.alfa 
    beta = args.beta
        
    try:
            
        ###
        ###  2. Lookup for k most relevant files
        ### 
        
        client = Elasticsearch()
        search = Search(using=client, index=index)
        if query is not None:
            #Get all queries together doing a bitwise AND
            q = Q("query_string", query=query[0])
            for i in range(1, len(query)):
                q &= Q("query_string", query=query[i])
            
            #search in index for k documents
            search = search.query(q)
            results = search[0:k].execute()
            query_dicc = query_to_dictionary(query)

            print(f"\n\n Initial query, {query_dicc}")
            show_most_relevant_documents(results)
            print("\n")

            ###
            ###  3. Apply rocchio's rule
            ### 
            
            refined_query, relevant_documents = get_k_most_relevant_documents_from_rocchio_rule(
                search=search,
                client=client,
                index=index,
                alfa=alfa,
                beta=beta,
                nrounds=nrounds, 
                k=k,
                R=R,
                initial_query=query_dicc,
                initial_related_documents=results,
            )
        else:
            print("query is empty")
            
        print (f"{relevant_documents.hits.total['value']} Documents")
        
    #Falta controlar esta excepcion para que pare el flujo del programa y retornar los k documentos
    except NotFoundError:
        print(f"Index {index} does not exists")

    
    
    
    
    #Para el numero de rondas podria calcular la diferencia promedio entre cada peso 

    
    #NO ESTA AFECTANDO EL PESO TF-IDF POR ALGUN MOTIVO