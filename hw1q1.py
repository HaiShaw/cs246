#!/usr/bin/env python2
# Hai Xiao, SUNetID: haixiao, Email: haixiao@stanford.edu

import re
import sys
import itertools as it
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

# read in friend lines
# friendLines = sc.textFile('t.txt')
friendLines = sc.textFile('soc-LiveJournal1Adj.txt')

# split <User> and <Friends>, note: <Friends> could be Null 
user_friends = friendLines.map(lambda l: l.split())

# user rdd with <User> only, and build python User list for later use
user  = user_friends.map(lambda x: x[0])
Users = user.collect()
Users = map(int, Users)

# new rdd of only users with friends
user_w_friends = user_friends.filter(lambda e: len(e) == 2)

# new rdd of element (user, list_of_friends), all int type, list_of_friend is
# sorted to assure deterministic at creating pairs of (person1, person2) that
# both are friends of the user at later time
user_friendList = user_w_friends.map(lambda p: (int(p[0]), map(int, sorted(p[1].split(',')))))


# N.B. - important MAP function
# This is the key MAP step of Map-Reduce process
# <K, V> it creates as:
#  K - (person1, person2), pair of person friended to user, person1 and 2 may be friend or not
#  V - 1, count to accumulate at Reduce phase
# Note: Need to filter out pairs who are already friends ~ Reduce at later time!
#       Also itertools.combinations keep tuple elements' order as in input list,
#       so has its input list sorted is necessary.
def friend_pairs_of_user(user_friendslist):
	from_user = user_friendslist[0]
	friendLst = user_friendslist[1]
	return [(pair_of_usersFriend, 1) for pair_of_usersFriend in it.combinations(friendLst, 2)]

pairs_from_commonFriend = user_friendList.map(friend_pairs_of_user).flatMap(lambda x: x)

# Now do Reduce function. (Filter pairs who already friended later)
# After this we have a pair of (person1,person2) as KEY, and their common friends count as VAL
pairs_CntOf_commonFriends = pairs_from_commonFriend.reduceByKey(lambda c1, c2: c1 + c2)

# Now to filter out pairs who were already friends directly
# I use a global map - user_friendList.collectAsMap() to test friendship of a pair, and filter.
# This may not be space efficient, it is at O(N), but much better than N x N space requirement.
# Afterwards, shall look at a RDD solution to take full advantage of parallization.

# user2friendsMap is a global Python dictionary.
user2friendsMap = user_friendList.collectAsMap() 

# Filter out pairs already directly friended
# Result is pairs of person who has common friends (with shared friends count), but yet friended
pairs_hasCommon_yetFriends = pairs_CntOf_commonFriends.filter(lambda pC: pC[0][1] not in user2friendsMap[pC[0][0]])


# Now transform a new rdd in the form of: (person_1, {commonFriendsCnt: [person_2]}) from pairs_hasCommon_yetFriends
# foreach ((person1, person2), cnt) ==create=two==> (person1, {cnt: [person2]}) and (person2, {cnt: [person1]})
user_recommendList_byShareCnt = pairs_hasCommon_yetFriends.map(lambda pC: [(pC[0][0], {pC[1]: [pC[0][1]]}), (pC[0][1], {pC[1]: [pC[0][0]]})]).flatMap(lambda x: x)
# sharedFriendsCnt = user_recommendList_byShareCnt


# Now we need to merge the recommendation list by shared friends count, by user.

# define custom merge function for reduce.
# d1, d2 are dictionay of form: {cnt: [...]}
def mergeByShareCnt(d1, d2):
	for k2 in d2:
		if k2 in d1:
			d1[k2] += d2[k2]
		else:
			d1[k2] = d2[k2]
	return d1

recommendList_byShareCnt = user_recommendList_byShareCnt.reduceByKey(mergeByShareCnt)


# Now sort each person's friend recommend list, by shared friends cnt (descending order) then friend candidate's User ID (ascending order)
# This is a requirement!

# define custom map function
# Input: (user, {cnt1: [...], cnt2: [...], ...})
# Output (user, [(cnt1, [..]), (cnt2, [.]), ..]), cnt - descending, [...] - ascending
def sortRecommendsByCnt(recommendTuple):
	rlist = []
	rdict = recommendTuple[1]
	ckeys = sorted(rdict, reverse=True)
	for cnt in ckeys:
	    rlist.append((cnt, sorted(rdict[cnt])))
	return (recommendTuple[0], rlist)

recommends = recommendList_byShareCnt.map(sortRecommendsByCnt)


# Now create new RDD, convert from
# (user, [(cnt1, [..]), (cnt2, [.]), ..])
# To form of merged list in orders
# (user, [...])
def recList(pair):
	person = pair[0]
	cntfList = pair[1]
   	cntfLen = len(cntfList)
   	rlist = []
	for i in range(cntfLen):
		for j in range(len(cntfList[i][1])):
			rlist.append(cntfList[i][1][j])
	return (person, rlist)


recommendList = recommends.map(recList)
# = rPtr

# Build Python dictionary for output /print
# form: {user1: [rec1, rec2, ...], ...}
recommendMap = recommendList.collectAsMap()
# printMap

# Define print function
# pMap - map to print
# pList - list of keys
#     N - upto N per key
# fname - file print to
def pMap2file(pMap, pList, N, fname):
	fp = open(fname, 'w+')
	# pList = sorted(pMap.keys())
	for k in pList:
		if k in pMap:
			plen = min(len(pMap[k]), N)
			s = ','.join(str(e) for e in pMap[k][:plen])
			print >> fp, k, '\t', s
		else:
			print >> fp, k
	fp.close()


uList = sorted(Users)
pMap2file(recommendMap, uList, 10, 'output_all.txt')

uList = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
pMap2file(recommendMap, uList, 10, 'select_out.txt')

sc.stop()
