#!/usr/bin/env python2
# Hai Xiao, SUNetID: haixiao, Email: haixiao@stanford.edu
# Iterative k-means cluster algorithm (Spark implementation)

import sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext

FLOATMAX = sys.float_info.max
MAX_ITER = 20
MID_ITER = 10

# First - Id each centroid, so to use it's id (to be in rdd) as key if needed
#         Don't have to do this if use tuple of 10 as key, but Id is simplier

# random centeroids
with open('c1.txt', 'r') as random_cents:
    c1lines = random_cents.readlines()

with open('rCents.txt', 'w') as r_cents:
    for (num, line) in enumerate(c1lines):
        r_cents.write('%d:%s' % (num, line))

# departed centroids 
with open('c2.txt', 'r') as farest_cents:
    c2lines = farest_cents.readlines()

with open('dCents.txt', 'w') as d_cents:
    for (num, line) in enumerate(c2lines):
        d_cents.write('%d:%s' % (num, line))

# Init Spark
conf = SparkConf()
sc = SparkContext(conf=conf)

# Prepare initial cluster centroids that are random chosen (c1.txt)
initr_centroids = 'rCents.txt'
initr_cents = sc.textFile(initr_centroids).map(lambda l: l.split(':')).map(lambda p: (int(p[0]), map(lambda x: float(x), p[1].split())))

# numpy array the coordinates for computing performance
ra_cents = initr_cents.map(lambda p: (p[0], np.array(p[1])))
# [(0, array([   0.   ,    0.64 ,    0.64 ,    0.   ,    0.32 ,    0.   ,
#                0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.64 ,
#                0.   ,    0.   ,    0.   ,    0.32 ,    0.   ,    1.29 ,
#                1.93 ,    0.   ,    0.96 ,    0.   ,    0.   ,    0.   ,
#                0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                0.   ,    0.   ,    0.   ,    0.778,    0.   ,    0.   ,
#                3.756,   61.   ,  278.   ,    1.   ])), (1, array([...])), ...]


# initial centroids map, r_cents[i] to access coordinates as nparray object
r_cents = ra_cents.collectAsMap()
# r_cents[0] = array([   0.   ,    0.64 ,    0.64 ,    0.   ,    0.32 ,    0.   ,
#                        0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.64 ,
#                        0.   ,    0.   ,    0.   ,    0.32 ,    0.   ,    1.29 ,
#                        1.93 ,    0.   ,    0.96 ,    0.   ,    0.   ,    0.   ,
#                        0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                        0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                        0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                        0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,
#                        0.   ,    0.   ,    0.   ,    0.778,    0.   ,    0.   ,
#                        3.756,   61.   ,  278.   ,    1.   ])
# this cents map can be kept global on each nodes, since it can't be big.
# For comparison, we can't expect all the data points to be available on
# each node, they are supposed to be distributed & processed in parallel.
# Hint: shall NOT build a similar global map for all the data points!


# Prepare initial cluster centroids that are chosen far apart (c2.txt)
initd_centroids = 'dCents.txt'
initd_cents = sc.textFile(initd_centroids).map(lambda l: l.split(':')).map(lambda p: (int(p[0]), map(lambda x: float(x), p[1].split())))
da_cents = initd_cents.map(lambda p: (p[0], np.array(p[1])))
d_cents = da_cents.collectAsMap()


# RDD for data points, they are distributed, no need to Id them
data = sc.textFile('data.txt').map(lambda line: map(lambda x: float(x), line.split()))
# Numpy-ize point coordinates
points = data.map(lambda cl: np.array(cl))

# Iterative K-Means algorithm:
# Key MAP functions - generating new RDD: points_assignment (K, V) per data point input.
# K - intermediate closest centroids Id (base on previous centroids coordinates) by min_distance
# V - (1, nparray from coordinate of the data point, cost/error term incurred by the assignment)
# So that during REDUCE phase: we can aggregate the total points count - N, coordinates.sum() of
# all assignee points, and total cost/error to each iterative cluster. Then new centroids can be
# computed from spatial mean in hyperspace, iteration continues to a lower cost / error attained.

# Find closest centeroid using Euclidean distance, this is the main MAP function
# Input: the point's coordinates (hyperspace) vector in numpy array
# Return:  (K, V)
#			K - Id of the closest centroid to this point (cluster be assigned)
#			V - (1, point's coordinate, cost/error iterm from this assignment)
def closest_euc(point):
	minDist = FLOATMAX
	cid = None
	for i in cents.keys():
		# use l2-norm from numpy linalg norm
		dist = np.linalg.norm(point-cents[i])
		if dist < minDist:
			minDist = dist
			cid = i
    # cost/error term introduced by single point
	cost = minDist * minDist
	return (cid, (1, point, cost))


# Find closest centroid with min distance, MAP function when use Manhattan distance
# Input: the point's coordinates (hyperspace) vector in numpy array
# Return:  (K, V)
#			K - Id of the closest centroid to this point (cluster be assigned)
#			V - (1, point's coordinate, cost/error iterm from this assignment)
def closest_man(point):
	minDist = FLOATMAX
	cid = None
	for i in cents.keys():
		# use l1-norm, it is same as:
        # dist = np.abs(point-cents[i]).sum()
		dist = np.linalg.norm(point-cents[i], ord=1)
		if dist < minDist:
			minDist = dist
			cid = i
    # cost/error term introduced by single point
	cost = minDist
	return (cid, (1, point, cost))


# Main REDUCE function
# Aggregate: the total points count - N, coordinates.sum() of all assigned points,
# and total cost/error to the last cluster center (Id as Key) from this iteration.
# Later, new centroids can be computed, and iterations continue.
def points_aggregate(p1, p2):
	n = p1[0] + p2[0]
	coordinates_sum = p1[1] + p2[1]
	cost_per_center = p1[2] + p2[2]
	return (n, coordinates_sum, cost_per_center)

# Records of costs from iterations
r_euc_costs = []
d_euc_costs = []
r_man_costs = []
d_man_costs = []

# Main Iterations - Euclidean Distance, Random Init Centroids
cents = r_cents
for i in range(MAX_ITER):
	# Assign points to the cluster with the closest centroids
	points_assignment = points.map(closest_euc)

	# Gather points cloud and statistic per cluster
	clusters = points_assignment.reduceByKey(points_aggregate)

	# Compute and update new centroids
    # New centroid coordinates = sum(coordinates_of_points_cloud) / N
	new_centers = clusters.map(lambda pc: (pc[0], pc[1][1]/pc[1][0]))
	cents = new_centers.collectAsMap()

	# Record cost/error value of this iteration
	total_costs = clusters.map(lambda pc: pc[1][2]).sum()
	r_euc_costs.append(total_costs)

# Main Iterations - Euclidean Distance, Far-apart Init Centroids
cents = d_cents
for i in range(MAX_ITER):
	# Assign points to the cluster with the closest centroids
	points_assignment = points.map(closest_euc)

	# Gather points cloud and statistic per cluster
	clusters = points_assignment.reduceByKey(points_aggregate)

	# Compute and update new centroids
	new_centers = clusters.map(lambda pc: (pc[0], pc[1][1]/pc[1][0]))
	cents = new_centers.collectAsMap()

	# Record cost/error value of this iteration
	total_costs = clusters.map(lambda pc: pc[1][2]).sum()
	d_euc_costs.append(total_costs)

# Main Iterations - Manhattan Distance, Random Init Centroids
cents = r_cents
for i in range(MAX_ITER):
	# Assign points to the cluster with the closest centroids
	points_assignment = points.map(closest_man)

	# Gather points cloud and statistic per cluster
	clusters = points_assignment.reduceByKey(points_aggregate)

	# Compute and update new centroids
	# N.B. Spatial specialty of Manhattan Space:
	# i.e. No (infinite) shortest path between 2 points!
	# mean may not be the best way => new centroid (Manhattan Space)
	new_centers = clusters.map(lambda pc: (pc[0], pc[1][1]/pc[1][0]))
	cents = new_centers.collectAsMap()

	# Record cost/error value of this iteration
	total_costs = clusters.map(lambda pc: pc[1][2]).sum()
	r_man_costs.append(total_costs)

# Main Iterations - Manhattan Distance, Far-apart Init Centroids
cents = d_cents
for i in range(MAX_ITER):
	# Assign points to the cluster with the closest centroids
	points_assignment = points.map(closest_man)

	# Gather points cloud and statistic per cluster
	clusters = points_assignment.reduceByKey(points_aggregate)

	# Compute and update new centroids
	# N.B. Spatial specialty of Manhattan Space:
	# i.e. No (infinite) shortest path between 2 points!
	# mean may not be the best way => new centroid (Manhattan Space)
	new_centers = clusters.map(lambda pc: (pc[0], pc[1][1]/pc[1][0]))
	cents = new_centers.collectAsMap()

	# Record cost/error value of this iteration
	total_costs = clusters.map(lambda pc: pc[1][2]).sum()
	d_man_costs.append(total_costs)

fp = open('hw2q2-output.txt', 'w+')

print >> fp, "k-Means cluster Costs: (Euclidean Distance, Random Init Centroids, k = 10, ITER = 20)"
print >> fp, "Cost (after 10 iterations) changed to: %.2f%s\tDropped: %.2f%s" % (100*r_euc_costs[9]/r_euc_costs[0], '%', 100*(1-r_euc_costs[9]/r_euc_costs[0]), '%')
print >> fp, r_euc_costs
print >> fp

print >> fp, "k-Means cluster Costs: (Euclidean Distance, Far-apart Init Centroids, k = 10, ITER = 20)"
print >> fp, "Cost (after 10 iterations) changed to: %.2f%s\tDropped: %.2f%s" % (100*d_euc_costs[9]/d_euc_costs[0], '%', 100*(1-d_euc_costs[9]/d_euc_costs[0]), '%')
print >> fp, d_euc_costs
print >> fp

print >> fp, "k-Means cluster Costs: (Manhattan Distance, Random Init Centroids, k = 10, ITER = 20)"
print >> fp, "Cost (after 10 iterations) changed to: %.2f%s\tDropped: %.2f%s" % (100*r_man_costs[9]/r_man_costs[0], '%', 100*(1-r_man_costs[9]/r_man_costs[0]), '%')
print >> fp, r_man_costs
print >> fp

print >> fp, "k-Means cluster Costs: (Manhattan Distance, Far-apart Init Centroids, k = 10, ITER = 20)"
print >> fp, "Cost (after 10 iterations) changed to: %.2f%s\tDropped: %.2f%s" % (100*d_man_costs[9]/d_man_costs[0], '%', 100*(1-d_man_costs[9]/d_man_costs[0]), '%')
print >> fp, d_man_costs
print >> fp

fp.close()

# Graph plot function:
# color: 'bs-', 'g^-', 'ro-'
def plot_fig(xlabel, Xs, ylabel, title, Y1s, color1, legend1, Y2s, color2, legend2, filename = None):
    plt.figure(figsize=(12,6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(Xs)
    plt.plot(Xs, Y1s, color1, label=legend1)
    plt.plot(Xs, Y2s, color2, label=legend2)
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(filename)
        plt.close(filename)
    else:
        plt.show()


X = range(1, MAX_ITER+1)

# Graph plot of (a).1
plot_fig('X axis - Iteration (MAX_ITER=20)', X, 'Y axis - Cost (cluster count k = 10)', \
         'k-Means iterative cluster (Euclidean distance): value of cost functions vs. iterations', \
         r_euc_costs, 'bs-', 'centroids init random (c1.txt)', d_euc_costs, 'g^-', 'centroids init farapart (c2.txt)', 'hw2q2-a.png')

# Graph plot of (b).1
plot_fig('X axis - Iteration (MAX_ITER=20)', X, 'Y axis - Cost (cluster count k = 10)', \
         'k-Means iterative cluster (Manhattan distance): value of cost functions vs. iterations', \
         r_man_costs, 'bs-', 'centroids init random (c1.txt)', d_man_costs, 'g^-', 'centroids init farapart (c2.txt)', 'hw2q2-b.png')

sc.stop()

