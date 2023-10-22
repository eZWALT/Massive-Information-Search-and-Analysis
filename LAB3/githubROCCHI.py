"""
.. module:: SearchIndexWeight

SearchIndex
*************

:Description: SearchIndexWeight

    Performs a AND query for a list of words (--query) in the documents of an index (--index)
    You can use word^number to change the importance of a word in the match

    --nhits changes the number of documents to retrieve

:Authors: bejar
    

:Version: 

:Created on: 04/07/2017 10:56 

"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
from elasticsearch.client import CatClient

import numpy as np
import operator

__author__ = 'bejar'

"""
PREVIOUS LAB FUNCTIONS
"""

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])

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
        raise NameError('File [' + path + '] not found');
    else:
        return lfiles[0].meta.id

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
    termvector = client.termvectors(index=index, id=id, fields=['text'], doc_type=['_doc'],
                                    positions=False, term_statistics=True, )

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())

def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    norm = 0;
    twRet = {};

    for (_, tfidf) in tw.items():
        norm = norm + tfidf**2;

    norm = np.sqrt(norm);

    for (t, tfidf) in tw.items():
        #print(type(t), type(tfidf), type(norm))
        if norm == 0:
            twRet[t] = 0.0
        else: 
            twRet[t] = float(tfidf)/float(norm);

    return twRet;

def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get document terms frequency and overall terms document frequency
    # [(term, freq in doc)] [(term, freq in index)]
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv]) # max f in doc

    dcount = doc_count(client, index)   # num of docs

    tfidfw = {}
    for (t, w),(_, df) in zip(file_tv, file_df):
        tf  = w / max_freq;
        idf = np.log2((dcount / df));
        tfidfw[t] = tf * idf;

    #print('ABANS NORMALIZE ', tfidfw)
    return normalize(tfidfw) # [(term, TFIDF)]

def QtoD(query):
    dict_result = {}
    for elem in query:
        if '^' in elem:
            word, value = elem.split('^')
            value = int(value)
            dict_result[word] = value
        else:
            if('~' in elem):
                word, value = elem.split('~')
                value = 1
                dict_result[word] = value
            else:
                word = elem
                value = 1
                dict_result[word] = value
    #print('AAAA', dict_result)
    return (dict_result)

def DtoQ(dic):
    #print('dic\n', dic)
    query_result = []
    for word, value in dic.items():
        elem = word + '^' + str(value)
        query_result.append(elem)
    #print('BBBB', query_result)
    return query_result

def tfidfofDict(dict_query, reldocs, client, index):
    tfidf_result = {}

    for word,_ in dict_query.items():
        tfidf_result[word] = 0

    for doc in reldocs:
        id = search_file_by_path(client, index, doc.path)
        tfidf_doc = toTFIDF(client, index, id)
        # print(sorted(tfidf_doc.items(), key=operator.itemgetter(1)))
        # for key, value in tfidf_doc.items():
        #     if (value != 0.0):
        #         print(key,value)
        
        for word,_  in dict_query.items():
            tfidf_result[word] += tfidf_doc[word]

    #print('tfidf result', dict_query)
    return tfidf_result

def RocchioRule(alpha,beta,dict_query,tfidf_dict,nhits):
    query= {}
    for word, weight in dict_query.items():
        #print('!!!!', tfidf_dict[word], nhits)
        meank = tfidf_dict[word]/nhits
        new = float(alpha)*float(weight) + float(beta)*float(meank)
        query[word] = new
    return query

"""
MAIN 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nhits', default=10, type=int, help='Number of hits to return')
    parser.add_argument('--nrounds', default=1, type=int, help='Iterations of Rocchio Law formula')
    parser.add_argument('-R', default=None, type=int, help="Maximum number of new terms to be kept in q'")
    parser.add_argument('--alpha', default=None, type=int, help='Alpha')
    parser.add_argument('--beta', default=None, type=int, help='Beta')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    query = args.query
    #print(query)
    nhits = args.nhits
    nrounds = args.nrounds
    r = args.R
    alpha = args.alpha
    beta = args.beta
    
    #1. obte diccionari a partir de query 

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)

        #=================#
        # PARAMS          # k most relevant documents 
        #=================#

        if query is not None:
            q = Q('query_string',query=query[0])
            for i in range(1, len(query)):
                q &= Q('query_string',query=query[i])

            s = s.query(q)
            response = s[0:nhits].execute()

            # Query to Dictionary in the shape [word, weight]
            dict_query = QtoD(query)
            print('i = 0, ' + 'q = ' + str(query))
            tfidfw_query = tfidfofDict(dict_query,response,client,index)


            #=================#
            # ROCCHIO RULE    #
            #=================#
            for i in range(2,nrounds):
                # Rocchio returns our new query
                #print('abans de Rocchio', dict_query)
                dict_query = RocchioRule(alpha, beta, dict_query, tfidfw_query, nhits)
                #print('despres de Rocchio', dict_query)
                new_query = DtoQ(dict_query)
                print('i = ' + str(i-1) + ', q = ' + str(new_query))

                q = Q('query_string',query=new_query[0])
                for i in range(1, len(query)):
                    q &= Q('query_string',query=new_query[i])

                s = s.query(q)
                response = s[0:nhits].execute()
                tfidfw_query = tfidfofDict(dict_query, response, client, index)

            #=================#
            # LAST QUERY      # and retrieve result
            #=================#
            dict_query = RocchioRule(alpha, beta, dict_query, tfidfw_query, nhits)
            #print(dict_query)
            new_query = DtoQ(dict_query)
            print('i = ' + str(nrounds-1) + ', q = ' + str(new_query))


            q = Q('query_string',query=new_query[0])
            for i in range(1, len(query)):
                q &= Q('query_string',query=new_query[i])

            s = s.query(q)
            response = s[0:nhits].execute()

            for r in response:  # only returns a specific number of results
                print('ID=' + r.meta.id + ' SCORE=' + str(r.meta.score))
                print('PATH=' + r.path)
                print('TEXT: ' + r.text[:50])
                print('-----------------------------------------------------------------')

        else:
            print('No query parameters passed')

        print (str(response.hits.total['value']) + " Documents")

    except NotFoundError:
        print('Index ' + str(index) + ' does not exists')

