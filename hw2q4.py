#!/usr/bin/env python2
# Hai Xiao, SUNetID: haixiao, Email: haixiao@stanford.edu
# Recommendation Systems - User-User & Item-Item Collaborative Filtering

import sys
import numpy as np

# load TV shows from file
shows = []
with open('shows.txt', 'r') as f:
	for showline in f:
		shows.append(showline.strip()[1:-1])

# load ratings Matrix R (m x n numpy matrix) from file 
R = np.loadtxt('user-shows.txt', dtype='int')

# calculate axis sum

# m = 9985 (user counts), n = 563 (movie counts)
m, n = R.shape

# init diagonal Matrix P (m x m)
P = np.zeros(shape=(m, m), dtype='int')

# init diagonal Matrix Q (n x n)
Q = np.zeros(shape=(n, n), dtype='int')

# calculate R axis sum to fill P, Q
# R_s0 (563,) to fill Q
R_s0 = np.sum(R, axis=0)
# R_s1 (9985,) to fill P
R_s1 = np.sum(R, axis=1)

# each user
for u in range(m):
	P[u][u] = R_s1[u]

# each item/movie
for i in range(n):
	Q[i][i] = R_s0[i]

# P, Q is done computation

# R's transpose
Rt = np.transpose(R)

# Now use R, Rt, P, Q
# to compute User Similarity Matrix Su; and Item Similarity Matrix Si as:
# 
#       -1/2     T   -1/2
# Su = P     R  R   P
#
#      -1/2   T      -1/2
# Si = Q     R  R   Q

RtR = np.matmul(Rt, R)
RRt = np.matmul(R, Rt)

#  -1/2
# Q      - Q's sqrt's inverse
Q_1o2 = np.linalg.inv(np.sqrt(Q))

#  -1/2
# P      - P's sqrt's inverse
P_1o2 = np.linalg.inv(np.sqrt(P))

# m x m
Su_tmp = np.matmul(P_1o2, RRt)
Su = np.matmul(Su_tmp, P_1o2)

# n x n
Si_tmp = np.matmul(Q_1o2, RtR)
Si = np.matmul(Si_tmp, Q_1o2)

# Now compute Recommendation (score) Matrix Gamma Y
# Yu: for user-user collaborative filtering
# Yu = Su x R
Yu = np.matmul(Su, R)

# Yi: for item-item collaborative filtering
# Yi = R x Si
Yi = np.matmul(R, Si)

# Now get recommendations from the first 100 shows for Alex (user index = 499)
# pick top5 of highest similarity scores, in case of ties, choose lower index

# recommendations user-user collaborative filter
rec_u2u_cf = (-Yu[499][0:100]).argsort()[:5]

# recommendations item-item collaborative filter
rec_i2i_cf = (-Yi[499][0:100]).argsort()[:5]


fp = open('hw2q4-output.txt', 'w+')

print "Using user-user collaboration, TOP 5 recommendations from the first 100 shows for Alex:"
print >> fp, "Using user-user collaboration, TOP 5 recommendations from the first 100 shows for Alex:"
for i in range(5):
	print "\tSimilarity score = %.5f" % (Yu[499][rec_u2u_cf[i]]), "\tMovie: ", shows[rec_u2u_cf[i]]
	print >> fp, "\tSimilarity score = %.5f" % (Yu[499][rec_u2u_cf[i]]), "\tMovie: ", shows[rec_u2u_cf[i]]

print
print >> fp

print "Using item-item collaboration, TOP 5 recommendations from the first 100 shows for Alex:"
print >> fp, "Using item-item collaboration, TOP 5 recommendations from the first 100 shows for Alex:"
for i in range(5):
	print "\tSimilarity score = %.5f" % (Yi[499][rec_i2i_cf[i]]), "\tMovie: ", shows[rec_i2i_cf[i]]
	print >> fp, "\tSimilarity score = %.5f" % (Yi[499][rec_i2i_cf[i]]), "\tMovie: ", shows[rec_i2i_cf[i]]

fp.close()
