#!/usr/bin/env python2
# Hai Xiao, SUNetID: haixiao, Email: haixiao@stanford.edu
# A-priori algorithm using Triples methods with distributed Hash (Spark <K, V>) Implementation

import re
import sys
import itertools as it
from pyspark import SparkConf, SparkContext

# First - Reformat Input file 'browsing.txt' by adding lineNumber as basketId
#         at beginning of each line, output to 'browsingBasket.txt' then feed
#         to Spark textFile() API for post processing.

with open('browsing.txt', 'r') as browsing:
	lines = browsing.readlines()

with open('browsingBasket.txt', 'w') as bbrowsing:
	for (num, line) in enumerate(lines):
		bbrowsing.write('%d:%s' % (num, line))

conf = SparkConf()
sc = SparkContext(conf=conf)

basklines = sc.textFile('browsingBasket.txt')
basket_Items = basklines.map(lambda l: l.split(':'))

# new RDD of shape: [(0, [u'ELE17451', u'ELE89019', u'FRO11987', u'GRO99222', u'SNA90258']), ...]
basketItems = basket_Items.map(lambda p: (int(p[0]), sorted(p[1].split())))

# MAP function to find frequent items
def itemsOf_baskets(basket):
	basketId = basket[0]
	itemList = basket[1]
	return [(item, 1) for item in itemList]

items = basketItems.map(itemsOf_baskets).flatMap(lambda x: x)

supportThresh = 100

# REDUCE function to find frequent items
itemCounts = items.reduceByKey(lambda c1, c2: c1 + c2)

# new frequent_items RDD of shape: [(u'ELE11909', 131), (u'SNA91554', 208), ...]
frequent_items = itemCounts.filter(lambda x: x[1] >= supportThresh)

# Construct Hash Map of frequent Items for later use
freqItems = frequent_items.collectAsMap()

# Next:=> use this and upcoming global Maps to filter frequent singleton, doubleton, tripleton per basket
# Next:=> frequent singleton, doubleton, tripleton

# Use it to filter each basket => baskets of only frequent items called: basket_freqSingleton

def frequentSingletons(basket):
	basketId = basket[0]
	itemList = basket[1]
	freqList = []
	for item in itemList:
		if item in freqItems:
			freqList.append(item)
	return (basketId, sorted(freqList))

# baskets with only frequent items (i.e. SUPPORT >= 100) kept
basket_freqSingletons = basketItems.map(frequentSingletons)


# Now generate Candidate pairs (X, Y) from basket_freqSingletons. Hint: per basket, use combinations

def item_pairs_of_basket(basket):
	basketId = basket[0]
	itemList = basket[1]
	return [(pair_of_items, 1) for pair_of_items in it.combinations(itemList, 2)]

# Candidate Construct Step of A-prior algorithm, each item of frequent doubleton is also frequent!
candidate_doubles = basket_freqSingletons.map(item_pairs_of_basket).flatMap(lambda x: x)
doubleCounts = candidate_doubles.reduceByKey(lambda c1, c2: c1 + c2)

# Candidate Filter Step
frequent_pairs = doubleCounts.filter(lambda x: x[1] >= supportThresh)

# Construct Hash Map of frequent Pairs for later use
freqPairs = frequent_pairs.collectAsMap()


# Now up to generate basket_freqDoubletons, i.e. backet with only frequent doubletons kept! 

def frequentDoubletons(basket_freqSingles):
	basketId = basket_freqSingles[0]
	frequent = basket_freqSingles[1]
	pairList = []
	CandList = [pC for pC in it.combinations(frequent, 2)]
	for pair in CandList:
		if pair in freqPairs:
			pairList.append(pair)
	return (basketId, sorted(pairList))

basket_freqDoubletons = basket_freqSingletons.map(frequentDoubletons)


# Now up to generate Candidate triples (X, Y, Z) from basket_freqDoubletons
# Trick: recover a list of frequent items from frequent pairs per each basket! Hint: then per basket, use combinations of 3

def item_triples_of_basket(basket_freqPairs):
	basketId   = basket_freqPairs[0]
	freq_pairs = basket_freqPairs[1]
	freq_items = []
	for p in freq_pairs:
		i1, i2 = p
		if i1 not in freq_items:
			freq_items.append(i1)
		if i2 not in freq_items:
			freq_items.append(i2)
	return [(triples, 1) for triples in it.combinations(sorted(freq_items), 3)]

# Candidate Step
candidate_triples = basket_freqDoubletons.map(item_triples_of_basket).flatMap(lambda x: x)

# Candidate Filter Step 
tripleCounts = candidate_triples.reduceByKey(lambda c1, c2: c1 + c2)
frequent_triples = tripleCounts.filter(lambda x: x[1] >= supportThresh)

# Construct Hash Map of frequent Triples for later use
# As: {(u'DAI62779', u'DAI83733', u'ELE92920'): 103, (u'DAI75645', u'FRO40251', u'GRO94758'): 118, ...}
freqTriples = frequent_triples.collectAsMap()


# With 3 Hash Maps: freqItems, freqPairs, freqTriples
# Up to calculate associate rule in descending order!

# pairs - freqPairs
# items - freqItems
def rules_fromPairs_sortByConf(pairs, items):
	ruleList = []
	for pair in pairs:
		x = pair[0]
		y = pair[1]
		X_support = items[x]
		Y_support = items[y]
		XYsupport = pairs[pair]
		Conf_XtoY = (1.0 * XYsupport)/X_support
		Conf_YtoX = (1.0 * XYsupport)/Y_support
		ruleList.append((Conf_XtoY, (x, y)))
		ruleList.append((Conf_YtoX, (y, x)))
	return sorted(ruleList, key = lambda x: (-x[0], x[1]))

one2oneRules = rules_fromPairs_sortByConf(freqPairs, freqItems)

topN = 5
for i in range(topN):
	print "Conf: %.8f\tRule: %s => %s" % (one2oneRules[i][0], one2oneRules[i][1][0], one2oneRules[i][1][1])
	"""
	Conf: 1.00000000	Rule: DAI93865 => FRO40251
	Conf: 0.99917628	Rule: GRO85051 => FRO40251
	Conf: 0.99065421	Rule: GRO38636 => FRO40251
	Conf: 0.99056604	Rule: ELE12951 => FRO40251
	Conf: 0.98672566	Rule: DAI88079 => FRO40251
	"""

print "\n\n"

# triples - freqTriples
#   pairs - freqPairs
def rules_fromTriples_sortByConf(triples, pairs):
	ruleList = []
	for triple in triples:
		x = triple[0]
		y = triple[1]
		z = triple[2]
		XY_support = pairs[(x, y)]
		XZ_support = pairs[(x, z)]
		YZ_support = pairs[(y, z)]
		XYZsupport = triples[triple]
		Conf_XYtoZ = (1.0 * XYZsupport)/XY_support
		Conf_XZtoY = (1.0 * XYZsupport)/XZ_support
		Conf_YZtoX = (1.0 * XYZsupport)/YZ_support
		ruleList.append((Conf_XYtoZ, ((x, y), z)))
		ruleList.append((Conf_XZtoY, ((x, z), y)))
		ruleList.append((Conf_YZtoX, ((y, z), x)))
	return sorted(ruleList, key = lambda x: (-x[0], x[1]))

two2oneRules = rules_fromTriples_sortByConf(freqTriples, freqPairs)

for i in range(topN):
	print "Conf: %.8f\tRule: (%s, %s) => %s" % (two2oneRules[i][0], two2oneRules[i][1][0][0], two2oneRules[i][1][0][1], two2oneRules[i][1][1])
	"""
	Conf: 1.00000000	Rule: (DAI23334, ELE92920) => DAI62779
	Conf: 1.00000000	Rule: (DAI31081, GRO85051) => FRO40251
	Conf: 1.00000000	Rule: (DAI55911, GRO85051) => FRO40251
	Conf: 1.00000000	Rule: (DAI62779, DAI88079) => FRO40251
	Conf: 1.00000000	Rule: (DAI75645, GRO85051) => FRO40251
	"""

# Define print function
# rList - list of rules
#     N - prn upto topN
# fname - file print to
def pOne2OneRules2file(rList, N, fname):
	fp = open(fname, 'w+')
	for i in range(N):
		print >> fp, "Conf: %.8f\tRule: %s => %s" % (rList[i][0], rList[i][1][0], rList[i][1][1])
	fp.close()

def pTwo2OneRules2file(rList, N, fname):
	fp = open(fname, 'w+')
	# pList = sorted(pMap.keys())
	for i in range(N):
		print >> fp, "Conf: %.8f\tRule: (%s, %s) => %s" % (rList[i][0], rList[i][1][0][0], rList[i][1][0][1], rList[i][1][1])
	fp.close()

# Lastly print complete associative rules to files

pOne2OneRules2file(one2oneRules, len(one2oneRules), '1-to-1-rules.txt')
pTwo2OneRules2file(two2oneRules, len(two2oneRules), '2-to-1-rules.txt')

sc.stop()
