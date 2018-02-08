#!/usr/bin/env python2
# Hai Xiao, SUNetID: haixiao, Email: haixiao@stanford.edu

from scipy import linalg
import numpy as np

# initialize M as (4x2) matrix
M = np.array([[1,2],[2,1],[3,4],[4,3]])

# Q1e.1 - SVD, calculate U, s, Vt

# run svd decomposition of M
U, s, Vt = linalg.svd(M, full_matrices=False)
print "U.shape, Sigma.shape, Vt.shape"
print U.shape, "    ", s.shape, "    ", Vt.shape
print
print "U:"
print U

# array([[-0.27854301,  0.5       ],
#        [-0.27854301, -0.5       ],
#        [-0.64993368,  0.5       ],
#        [-0.64993368, -0.5       ]])
print

print "Sigma:"
print s

# array([ 7.61577311,  1.41421356])
print

print "Vt"
print Vt

# array([[-0.70710678, -0.70710678],
#        [-0.70710678,  0.70710678]])
print


# Q1e.2 - engivalues/engivectors of Mt x M

# M's transpose - Mt
Mt = np.matrix.transpose(M)
print "M ="
print M
print
print "Mt (M's transpose) ="
print Mt
print

# MtxM
MtM = np.matmul(Mt, M)
print "Mt x M ="
print MtM

# array([[30, 28],
#        [28, 30]])
print

# calculate eigenvalues/eigenvectors of Mt x M
evals, evecs = linalg.eigh(MtM)
print "evals.shape, evecs.shape"
print evals.shape, "         ", evecs.shape
print
print "evals ="
print evals

# array([  2.,  58.])
print

print "evecs ="
print evecs

# array([[-0.70710678,  0.70710678],
#        [ 0.70710678,  0.70710678]])
print

# find indices of eigenvalues in descending order 
descendIndices = np.argsort(-evals)

# print descendIndices
# array([1, 0])

# Eigenvalues (reordered)
Eigenvalues = evals[descendIndices]
print "Evals (descending) ="
print Eigenvalues

# array([ 58.,   2.])
print

# Eigenvectors (reordered)
Eigenvectors = evecs[:,descendIndices]
print "Evecs (reordered) ="
print Eigenvectors

# array([[ 0.70710678, -0.70710678],
#        [ 0.70710678,  0.70710678]])
print

print "Happy mining massive dataset!\n"
