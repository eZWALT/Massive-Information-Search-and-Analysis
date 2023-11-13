#!/usr/bin/python

import time
import sys


class Edge:
    def __init__(self, origin=None, index = None):
        self.origin = origin  # write appropriate value
        self.weight = 1.0  # write appropriate value
        self.index = index

    def __repr__(self):
        return "edge: {0} {1}".format(self.origin, self.weight)

    ## write rest of code that you need for this class


class Airport:
    def __init__ (self, iden=None, name=None, index=None):
        self.code = iden				# IATA code
        self.name = name 				# Name of the airport
        self.routes = []				        # list of edges of the airports that have this airport as destination; list[edges]
        self.routeHash = dict()			# dict{key = airport IATA code : index in routes}
        self.outweight = 0.0			# weight from this airport to others

    def addEdge(self, origin):
        if origin in self.routeHash:
            idx = self.routeHash[origin]
            (self.routes[idx]).weight += 1.0
        else:
            edge = Edge(origin)
            edge.index = airportHash[origin].index
            i = len(self.routes)
            self.routes.append(edge)
            self.routeHash[origin] = i

    def __repr__(self):
        return f"{self.code}\t{self.index}\t{self.name}"


edgeList = []  # list of Edge
edgeHash = dict()  # hash of edge to ease the match
airportList = []  # list of Airport
airportHash = dict()  # hash key IATA code -> Airport
pageRank = []
nOut = 0


def readAirports(fd):
    print("Reading Airport file from {0}".format(fd))
    airportsTxt = open(fd, "r");
    cont = 0
    for line in airportsTxt.readlines():
        a = Airport()
        try:
            temp = line.split(',')
            if len(temp[4]) != 5:
                raise Exception('not an IATA code')
            a.name = temp[1][1:-1] + ", " + temp[3][1:-1]
            a.code = temp[4][1:-1]
            a.index = cont
        except Exception as inst:
            pass
        else:
            cont += 1
            airportList.append(a)
            airportHash[a.code] = a
    airportsTxt.close()
    print(f"There were {cont} Airports with IATA code")


def getAirport(code):
    if not (code in airportHash):
        raise Exception(f"Airport {code} not found")
    return airportList[airportHash[code].index]


def readRoutes(fd):
    print(f"Reading Routes file from {fd}")
    # write your code
    routesTxt = open(fd, "r")
    cont = 0
    for line in routesTxt.readlines():
        try:
            aux = line.split(',')

            if len(aux[2]) != 3 or len(aux[4]) != 3:
                raise Exception('not IATA')
            
            originCode = aux[2]
            destinationCode = aux[4]
            originAirport = getAirport(originCode)
            destinationAirport = getAirport(destinationCode)
            destinationAirport.addEdge(originCode)
            originAirport.outweight += 1.0

        except Exception as inst:
            pass
        else:
            cont += 1

    routesTxt.close()
    print(f"{cont} Airports with IATA code were found")

def computePageRanks():
    # write your code
    print("Computing Page Rank")
    n = len(airportList)
    P = [1.0 / n] * n
    L = 0.2
    iters = 0
    stop = False

    L1 = (1.0 - L) / n

    while not stop:
        Q = [0.0] * n
        for i in range(n):
            airport = airportList[i]
            summation = 0
            for edge in airport.routes:
                out = airportList[edge.index].outweight
                w = edge.weight
                summation += w * P[edge.index] / out

            Q[i] = L * summation + L1

        val = [x_i - y_i for x_i, y_i in zip(P, Q)]
        stop = all(map(lambda v: v < (1 * 10 ** (-14)), map(lambda v: abs(v), val)))
        #printSum(Q)
        P = Q
        iters += 1

    global pageRank
    pageRank = P
    return iters


def outputPageRanks():
    # write your code
    n = len(airportList)
    myList: dict = {key: p for key, p in zip(range(n), pageRank)}

    myList2 = sorted(myList.items(), key=lambda item: item[1], reverse=True)
    idxCount = 1
    for ap, pg in myList2:
        name = airportList[ap].name
        code = airportList[ap].code
        print(f"({pg}, {name})")
        idxCount += 1


def main(argv=None):
    readAirports("./airports.txt")
    readRoutes("./routes.txt")
    time1 = time.time()
    iterations = computePageRanks()
    time2 = time.time()
    outputPageRanks()
    print("#Iterations:", iterations)
    print("Time of computePageRanks():", time2 - time1)


if __name__ == "__main__":
    sys.exit(main())
