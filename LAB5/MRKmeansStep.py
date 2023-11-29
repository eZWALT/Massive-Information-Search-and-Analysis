"""
.. module:: MRKmeansDef

MRKmeansDef
*************

:Description: MRKmeansDef

    

:Authors: bejar
    

:Version: 

:Created on: 17/07/2017 7:42 

"""

from mrjob.job import MRJob
from mrjob.step import MRStep

__author__ = 'bejar'


class MRKmeansStep(MRJob):
    prototypes = {}

    def jaccard(self, prot, doc):
        """
        Compute here the Jaccard similarity between  a prototype and a document
        prot should be a list of pairs (word, probability)
        doc should be a list of words
        Words must be alphabeticaly ordered

        The result should be always a value in the range [0,1]
        """
        #sizes
        n = len(doc)
        m = len(prot)
        #indexes/pointers        
        i = 0
        j = 0
        
        #auxiliar variables
        dot_product = 0
        module_doc = 0
        module_prot = 0
        
        #compute dot product and part of the modules of both vectors
        while(i < n and j < m):
            term1 = doc[i]
            term2 = prot[j][0]
            weight1 = 1
            weight2 = prot[j][1]
            
            if term1 == term2:
                dot_product += weight1 * weight2 
                i += 1
                j += 1
                module_doc += weight1 * weight1
                module_prot += weight2 * weight2 
            elif term1 < term2: 
                i += 1
                module_doc += weight1 * weight1 
            else:
                j += 1 
                module_prot += weight2 * weight2 
                
        #compute the final bits of modules that do not belong to intersection
        while(i < n):
            weight = 1
            module_doc += weight * weight 
            i += 1
        
        while(j < m):
            weight = prot[j][1]
            module_prot = weight * weight 
            j += 1
        
        return dot_product / (module_doc + module_prot - dot_product)

    def configure_args(self):
        """
        Additional configuration flag to get the prototypes files

        :return:
        """
        super(MRKmeansStep, self).configure_args()
        self.add_file_arg('--prot')

    def load_data(self):
        """
        Loads the current cluster prototypes

        :return:
        """
        f = open(self.options.prot, 'r')
        for line in f:
            cluster, words = line.split(':')
            cp = []
            for word in words.split():
                cp.append((word.split('+')[0], float(word.split('+')[1])))
            self.prototypes[cluster] = cp

    #MAP FUNCTION
    def assign_prototype(self, _, line):
        """
        This is the mapper it should compute the closest prototype to a document

        Words should be sorted alphabetically in the prototypes and the documents

        This function has to return at list of pairs (prototype_id, document words)

        You can add also more elements to the value element, for example the document_id
        """

        # Each line is a string docid:wor1 word2 ... wordn
        doc, words = line.split(':')
        lwords = words.split()

        #The more similar they are the more close to 1 that number is
        similitude_of_closest = 0 
        closest_prototype = "CLASS0"
        for prototype_id, prototype_file in self.prototypes.items():
            new_similitude = self.jaccard(prototype_file, lwords)
            if new_similitude > similitude_of_closest:
                similitude_of_closest = new_similitude 
                closest_prototype = prototype_id

        # Return <cluster_id, document x>
        yield closest_prototype, lwords

    #REDUCE FUNCTION
    def aggregate_prototype(self, key, values):
        """
        input is cluster and all the documents it has assigned
        Outputs should be at least a pair (cluster, new prototype)

        It should receive a list with all the words of the documents assigned for a cluster

        The value for each word has to be the frequency of the word divided by the number
        of documents assigned to the cluster

        Words are ordered alphabetically but you will have to use an efficient structure to
        compute the frequency of each word

        :param key:
        :param values:
        :return:
        """
        num_docs = 0
        frequencies = {}
        average_inertia = 0
        
        values2 = []        
        for doc in values: 
            num_docs += 1
            values2.append(doc)
            for word in doc:
                if not word in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        new_prototype = []
        for word, freq in frequencies.items():
            new_prototype.append((word, freq/num_docs))
        
        #compute the average inertia(distortion)
        for doc in values2: 
            average_inertia += self.jaccard(new_prototype, doc) ** 2

        yield key, (new_prototype, average_inertia/float(num_docs))

    def steps(self):
        return [MRStep(mapper_init=self.load_data, mapper=self.assign_prototype,
                       reducer=self.aggregate_prototype)
            ]


if __name__ == '__main__':
    MRKmeansStep.run()