#!/usr/bin/python

from collections import namedtuple
import time
import sys
import argparse

edgeList = []           # list of Edge
edgeHash = dict()       # hash of edge to ease the match
airportHash = dict()    # hash key IATA code -> Airport

airportList = []        # list of all Airports
terminalList = []       # list of terminal airports
isolatedList = []       # list of isolated airports
unreachableList = []    # list of unreachable airports

PageRank = []

#Represents and edge (i,j,k) where i is the origin, j is the destination (implicit)
# and k is the weight (or number of routes)
class Edge:
    def __init__ (self, origin=None, index=None):
        self.origin = origin #origin airport
        self.index = index   #index on the routes list
        self.weight = 1.0    #every edge has weight 1 initially      

    def __repr__(self):
        return "edge: {0} {1}".format(self.origin, self.weight)
        
    ## write rest of code that you need for this class

class Airport:
    def __init__ (self, iden=None, name=None, index=None):

        #IATA code
        self.code = iden
        #Airport name
        self.name = name
        #edges of routes which end on this airport
        self.routes = []
        #stores (for incoming airports) IATA_code: index entries for fast retrieval on airport list
        self.routeHash = dict()
        #out weight of how many routes depart from this airport
        self.outweight = 0.0
        #index of airport in airport list (Optional) 
        self.index = index

    def add_edge(self, origin):
        if origin in self.routeHash:
            i = self.routeHash[origin]
            (self.routes[i]).weight += 1.0
        else:
            edge = Edge(origin)
            edge.index = airportHash[origin].index
            i = len(self.routes)
            self.routes.append(edge)
            self.routeHash[origin] = i

    def __repr__(self):
        return f"{self.code}\t{self.index}\t{self.name}"

def readAirports(fd):
    print("Reading Airport file from {0}".format(fd))
    airportsTxt = open(fd, "r") 
    cont = 0
    for line in airportsTxt.readlines():
        a = Airport()
        try:
            temp = line.split(',')
            #print(f"{temp[4]} {len(temp[4])}")
            if len(temp[4]) != 5 :
                raise Exception('not an IATA code')
            a.name=temp[1][1:-1] + ', ' + temp[3][1:-1]
            a.code=temp[4][1:-1]
            a.index = cont
        except Exception as inst:
            pass
        else:
            cont += 1
            airportList.append(a)
            airportHash[a.code] = a
    airportsTxt.close()
    print(f"There were {cont} Airports with IATA code")

def getAirport(iata_code):
    if not iata_code in airportHash:
        raise Exception(f"The airport {iata_code} was not found")
    return airportList[airportHash[iata_code].index]

def readRoutes(fd):
    print(f"Reading Routes file from {fd}")
    routes_txt = open(fd,"r")
    count = 0
    for line in routes_txt.readlines():
        try:
            fields = line.split(',')
            #IATA codes are located in the 3rd and 5th location
            if len(fields[2]) != 3 or len(fields[4]) != 3:
                raise Exception('not an IATA code')

            #Get the origin/destination airport IATA
            origin = fields[2]
            destination = fields[4]
            origin_airport = getAirport(origin)
            destination_airport = getAirport(destination)
            #add edge of incoming airport to departure airport
            destination_airport.add_edge(origin)
            #update weight of the origin airport (new edge)
            origin_airport.outweight += 1.0
            
        except Exception as inst:
            pass 
        else:
            count += 1
    routes_txt.close()
    print(f"There were {count} Routes with IATA code")

def eraseAirport(airport: Airport):
    iata = airport.code 
        


# O(n)
def getSpecialAirportsAndApplyRemoval(rem_isolated, rem_terminal, rem_unreachable):
    for airport in airportList:
        is_term = airport.outweight == 0.0
        is_isol = len(airport.routes)
        is_both = is_term and is_isol 

        if is_both:
            terminalList.append(airport)
            isolatedList.append(airport)
            unreachableList.append(airport)
            if rem_unreachable or rem_isolated or rem_terminal:
                eraseAirport(airport)

        elif is_isol:
            isolatedList.append(airport)
            if rem_isolated:
                eraseAirport(airport)

        elif is_term:
            terminalList.append(airport)
            if rem_terminal:
                eraseAirport(airport)


#OUT MUST BE COMPUTED AS THE SUM OF WEIGHTS OF ALL EDGES
#weight of an edge is all the repetitions of the same edge
def computePageRanks(Lambda: float, Epsilon: float):
    print("Computing ranking for all airports")
    n = len(airportList)
    #Initial distribution, each airport is as important as the rest
    P = [1.0/n for i in range(n)]
    #damping factor
    
    iterations = 0
    stop_condition = False

    while(not stop_condition):
        Q = [0.0 for i in range(n)]
        for i in range(n):
            #select i-th airport
            airport_i = airportList[i]
            summ = 0
            #iterate over all incoming routes of that airport
            for edge in airport_i.routes:
                summ += P[edge.index] * edge.weight / airportList[edge.index].outweight
            Q[i] = (Lambda * summ + (1.0 - Lambda)/n )

        iterations += 1
        Delta = [abs(x1-x2) for x1,x2 in zip(P,Q)]
        P = Q 
        stop_condition = all(map(lambda diff: diff < Epsilon , Delta))

    global PageRank
    PageRank = P  
    return iterations

def outputPageRanks(k):
    print(f"Displaying ranking of the first {k} airports and min-max values")
    print_info = {code: rank for code,rank in zip([a.code for a in airportList], PageRank)}
    print_info = sorted(print_info.items(), key=lambda item: item[1], reverse=True)

    min = print_info[-1]
    max = print_info[0]
    print(f"maximum value: {max} ----- minimum value: {min}")
    
    idx = 0
    while(idx < k):
        ap, pg = print_info[idx]
        print(f"({ap}: {pg})")
        idx += 1

def main(argv=None):

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", default=0.85, type=float, help="Damping factor for google matrix")
    parser.add_argument("--e", default=1 * 10**(-12), type=float, help="Epsilon > 0 small numbers used as a difference stopping condition in the PageRank loop")
    parser.add_argument("--k", default=10, type=int, help="Number of k first pageRanks to show")
    parser.add_argument("--t", default=False, type=bool, help="Boolean that indicates if terminal airports (no outgoing routes) are removed or not")
    parser.add_argument("--i", default=False, type=bool, help="Boolean that indicates if isolated airports (no incoming routes) are removed or not")
    parser.add_argument("--u", default=False, type=bool, help="Boolean that indicates if unreachable airports are removed or not")
    args = parser.parse_args()

    Lambda = args.l
    Epsilon = args.e 
    k = args.k 
    remove_isolated = args.i
    remove_terminal = args.t
    remove_unreachable = args.u
    
    #read files
    readAirports("./airports.txt")
    readRoutes("./routes.txt")
    getSpecialAirportsAndApplyRemoval(remove_isolated, remove_terminal, remove_unreachable)


    #basic main
    time1 = time.time()
    num_iterations = computePageRanks(Lambda, Epsilon)
    time2 = time.time()
    execution_time = time2-time1
    outputPageRanks(k)
    print("#Iterations:", num_iterations)
    print("Time of computePageRanks():", execution_time)


if __name__ == "__main__":
    sys.exit(main())
