#!/usr/bin/env python2
# Hai Xiao, SUNetID: haixiao, Email: haixiao@stanford.edu
# Latent Features for Recommendations - Stochastic Gradient Descent algorithm

import sys
import numpy as np
import matplotlib.pyplot as plt

"""
# Move away from below since it isn't expected to load all ratings into memory

import pandas as pd
ratings = pd.read_csv('ratings.train.txt', header=None, delim_whitespace=True)

# User Id range: [1, 943]
min_uid = ratings[0].min()
max_uid = ratings[0].max()

# Movie Id range: [1, 1682] 
min_mid = ratings[1].min()
max_mid = ratings[1].max()

# Ratings range: [1, 5]
min_rid = ratings[2].min()
max_rid = ratings[2].max()
"""

# ratings.train.txt is expected to be super big, e.g. 100 million ratings
# one way to find a field's max or min value from it is to use RDD Spark,
# straight fordward method is just to process lines sequentially as here:

max_id = max_uid = max_mid = max_rid = -1
min_id = min_uid = min_mid = min_rid =  1000

with open('ratings.train.txt', 'r') as f:
	for line in f:
		u,m,r = line.split()
		u = int(u)
		m = int(m)
		r = int(r)
		if u > max_uid:
			max_uid = u
		if u < min_uid:
			min_uid = u
		if m > max_mid:
			max_mid = m
		if m < min_mid:
			min_mid = m
		if r > max_rid:
			max_rid = r
		if r < min_rid:
			min_rid = r

print "User ID range:", min_uid, max_uid
print "Item ID range:", min_mid, max_mid
print "Ratings range:", min_rid, max_rid


# Factors dimension k = 20
k_factors = k = 20

# Random ceiling of init value to Q and P
max_init = (1.0* max_rid / k) ** 0.5

# Create and Initialize Parameter Matrices
#
# N.B. Assume Q and P can be loaded in mem
# i.e. Num of Users and Items isn't as big
# as an expected Number of Recommendations
# In case users / items counts are too big
# I will address in another implementation

# Pseudo random seed to repeatable results
# Items - factors Matrix: Q[1682 x 20]
np.random.seed(ord('q'))
Q = max_init * np.random.rand(max_mid, k)
print "Q (item-to-factor) Matrix shape:", Q.shape
print

# Users - factors Matrix: P[943 x 20]
np.random.seed(ord('p'))
P = max_init * np.random.rand(max_uid, k)
print "P (user-to-factor) Matrix shape:", P.shape

# P's Transpose
Pt = np.transpose(P)
print "P_t: P's transpose Matrix shape:", Pt.shape

# Save initial Q, P (for comparable tuning, on LR)
Q0 = 1*Q
P0 = 1*P

# Model Parameters
# lambda - the regularization term coefficient
ld = 0.1

# Training Parameters
# iterations
epochs = 40

# initial learning rate, to be tuned
lr = 0.036
LR = []

# list of objective function E values
# multiple lists for LR tuning
Ev = [[],[],[]]


# Main iterations of SGD

fp = open('hw2q3-output.txt', 'w+')
for lrt in range(len(Ev)):

	# Restore the same init Matrices. Deep copy!
	Q = 1*Q0
	P = 1*P0

	# lrt - learning rate tuning
	LR.append(lr)
	print
	print "Learning Rate = ", lr

	# list of Epoch counts
	Ep = []
	for ep in range(1, epochs+1):

		Ep.append(ep)

		# sgd processing of E_iu, Q_i (Q ith row) and P_u (P uth row) updates
		# using R_iu read from file line-by-line (R is too big to fit in mem)

		with open('ratings.train.txt', 'r') as f:
			for line in f:
				u,i,r = line.split()
				u = int(u)		# uid
				i = int(i)		# item
				# rating R_iu, N.B. we don't use float type,
				# use matrix R as original vs. Q|P as float.
				R_iu = int(r)

				# array indices
				u -= 1
				i -= 1
				# Scalar value. It is Derivative of error E wrt. R_iu
				E_iu = 2 * ( R_iu - np.dot(Q[i], P[u]) )

				# Update Q_i (ith row of Item-to-factor matrix Q)
				Q[i] = Q[i] + lr * ( E_iu * P[u] - 2 * ld * Q[i])

				# Update P_u (uth row of User-to-factor matrix P)
				P[u] = P[u] + lr * ( E_iu * Q[i] - 2 * ld * P[u])

		# Till now, finished one epoch of parameters sgd optimization

		# compute the objective function value Ev at the end of each iteration
		# using Q_i & P_u (i.e. updated Q, P matrices) from this iteration and
		# R_iu read from file line-by-line (R is too big to fit in host memory)

		# Init Error value
		E = 0.0

		# Init Q update Flag, per item: to update L2norm E term from Q[i] only if Q_uF[i] == 1
		Q_uF = [1 for i in range(max_mid)]

		# Init P update Flag, per user: to update L2norm E term from P[u] only if P_uF[u] == 1
		P_uF = [1 for i in range(max_uid)]

		with open('ratings.train.txt', 'r') as f:
			for line in f:
				u,i,r = line.split()
				u = int(u)    # user id
				i = int(i)    # movie id
				R_iu = int(r)

				# array indices
				u -= 1
				i -= 1

				# SSE Error term
				E += ( R_iu - np.dot(Q[i], P[u]) )**2

				# np.linalg.norm(a)**2 => L2 norm square,  but np.dot(a,a) is more efficient & accurate
				# N.B. sum up L2norm term only once to each index!
				if P_uF[u]:
					E += ld * np.dot(P[u], P[u])
					P_uF[u] = 0
				if Q_uF[i]:
					E += ld * np.dot(Q[i], Q[i])
					Q_uF[i] = 0

		# E is the sum of errors from all ratings
		Ev[lrt].append(E)
		print "\tEpoch = ", ep, "\t E = ", E

	print >> fp, "SGD of Latent Features Recommendadtions (epochs/iterations = 40, factors k = 20, L2 norm regularization lambda = 0.1, learning rate = %.3f)" % (lr)
	print >> fp, "SGD Epochs:"
	print >> fp, Ep
	print >> fp, "E (error value of objective function):"
	print >> fp, Ev[lrt]
	print >> fp
	# LR decay
	lr /= 2.

fp.close()

# main SGD iteration and LR tuning end this line

# Graph plot function (3 curves):
# color: 'ro-', 'bs-', 'g^-'
def plot_fig(xlabel, Xs, ylabel, title, Ys, colors, legends, filename = None):
    plt.figure(figsize=(12,6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(Xs)
    plt.plot(Xs, Ys[0], colors[0], label=legends[0])
    plt.plot(Xs, Ys[1], colors[1], label=legends[1])
    plt.plot(Xs, Ys[2], colors[2], label=legends[2])
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(filename)
        plt.close(filename)
    else:
        plt.show()

# Graph plot of Q3.(b)

legends = [[],[],[]]
for i in range(3):
	legends[i] = "Learning Rate = %.3f" % (LR[i])

plot_fig('X axis - Iteration (Epochs=40)', Ep, 'Y axis - Error Value of Objective Function (w. L2 Norm)', \
         'SGD optimization of Latent Features for Recommendations: Error vs. Epoch (diff learning rate)', \
         Ev, ['ro-', 'bs-', 'g^-'], legends, 'hw2q3.png')

