from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
import numpy as np
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search
from elasticsearch.client import CatClient
from elasticsearch_dsl.query import Q
from elasticsearch import Elasticsearch

import matplotlib.pyplot as plt
import plotille

import operator
import argparse
__author__ = 'walter'

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])

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

def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = {}
    for (term, w),(_, df) in zip(file_tv, file_df):
        #
        idfi = np.log2((dcount/df))
        tfdi = w/max_freq
        tfidfw[term] = tfdi * idfi
        # Something happens here
        #

    return normalize(tfidfw)

def normalize(document):
    summ = sum(document.values())
    sqrt = np.sqrt(summ)
    norm = {term: document.get(term, 0)/sqrt for term in set(document)}
    return norm

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

#
def get_dictionary_from_query(query):
    dQuery = {}
    for elem in query:

        if '^' in elem:
            key, val = elem.split('^')
            val = float(val)

        else:
            val = 1.0
            key = elem
        
        dQuery[key] = val
        
    return normalize(dQuery)

def get_query_from_dictionary(theDict):
    query = []

    for elem in theDict:
        q = elem + '^' + str(theDict[elem])
        query.append(q)
    
    return query

def rocchios_law(client, index, s, k, beta, alpha, R, nrounds , query):
    if query is not None:
        for iteration in range(0, nrounds):
            print(f"iteration number {iteration}")
            q = Q('query_string',query=query[0])
            for i in range(1, len(query)):
                q &= Q('query_string',query=query[i])

            print(query)
            s = s.query(q)
            response = s[0:k].execute()

            dict_query = get_dictionary_from_query(query)
            
            """
            if iteration == 0:
                x = 0
                for term,val in dict_query.items():
                    x = x + val 
                x = x/R
                nrounds_averages[iteration] = x
            #experimentació
            """
                
            merged_documents = {}
            
            #Convert all the K most relevant documents to tfidf dictionaries and merge them into 
            for r in response:
                file_tw = toTFIDF(client, index, r.meta.id) # tf-idf
                merged_documents = {term: merged_documents.get(term, 0) + file_tw.get(term, 0) for term in set(merged_documents) | set(file_tw)} # sumem els valors de cada document
                print(f'ID= {r.meta.id} SCORE={r.meta.score}')
                print(f'PATH= {r.path}')
                print(f'TEXT: {r.text[:50]}')
                print(f'ITERATION: {iteration}')
                print('-----------------------------------------------------------------')
                
            
            #Apply Rocchio's rule 
            merged_documents = {term: merged_documents.get(term,0)*beta/k for term in set(merged_documents)} # B * merged_documents / k
            old_query = {term: dict_query.get(term,0)*alpha for term in set(dict_query)} # a * query
            new_query = {}
            new_query = {term: merged_documents.get(term, 0) + old_query.get(term, 0) for term in set(merged_documents) | set(old_query)} # alpha * query + beta * merged_documents / K
            
            # sorterm and get the R most relevant terms, this can be done sorting or using priority queue in R*log(n) time
            new_query = sorted(new_query.items(), key=operator.itemgetter(1), reverse = True) 
            new_query = new_query[:R] 
            
            """
            x = 0
            for (term,val) in new_query:
                x = x + val 
            x = x/R
            #experimentacio 
            nrounds_averages[iteration] = x
            """
            # get query from dict
            dict_query = dict((term, val) for (term, val) in new_query) 
            
            
            query = get_query_from_dictionary(normalize(dict_query))
            print (f"{response.hits.total['value']} Documents")

    else:
        print('No query parameters passed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--k', default=10, type=int, help='Number of documents to return')
    parser.add_argument('--beta', default=1, type=float, help="beta coefficient of Rocchio's rule")
    parser.add_argument('--alpha', default=2, type=float, help="Alpha coefficient of Rocchio's rule")
    parser.add_argument('--R', default=5, type=int, help="Number of R most important terms of a document to use in document fusion")
    parser.add_argument('--nrounds', default=10, type=int, help="Number of times Rocchio's law is applied to the original query")
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')
    args = parser.parse_args()

    index = args.index
    k = args.k
    beta = args.beta
    alpha = args.alpha 
    R = args.R 
    nrounds = args.nrounds 
    query = args.query
    
    try:
        #Establish connection and search item
        client = Elasticsearch()
        s = Search(using=client, index=index)
        rocchios_law(client,index, s, k,beta,alpha,R,nrounds,query)

        
        
        """
        #Rolling means
        nrounds_averages = {}
        R_averages = {}
        alpha_averages = {}
        beta_averages = {}
        k_averages = {}
        
        for bbb in np.arange(0.01, beta, 0.5):
            #Apply Nrounds iterations of rocchios law of a given query
            nrounds_averages = {}
            rocchios_law(client,index, s, k,bbb,alpha,R,nrounds,query)
            beta_averages[bbb] = sum(nrounds_averages.values())/len(nrounds_averages)
        
        """
        #nrounds_averages = {}
        #rocchios_law(client,index, s, k,beta,alpha,R,nrounds,query)
        


            
        #Plot Nrounds average
        #fig = plotille.Figure()
        #print(plotille.plot(beta_averages.keys(), beta_averages.values(),X_label="Value of Beta",Y_label=f"Average of the average weigth after {nrounds} iteration", x_min=0, y_min = 0, x_max=beta, height=35, width=150, interp="linear", lc="green"))


    except NotFoundError:
        print(f'Index {index} does not exists')

 