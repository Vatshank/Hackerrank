###01/09/2014###
###Hackerrank; Category - STATS AND ML; Problem - THE BEST APTITUDE TEST; Difficulty - MODERATE###

from sklearn import linear_model
import numpy as np

##Get number of test cases
T = int(raw_input())
arr_result = np.zeros(T)

for t in range(T):
	test_mat = []
	##Get the number of students
	N = int(raw_input())
	##Get the GPA
	GPA = np.array(map(float,raw_input().strip().split()))
	##Need tp reshape 1-D array when using the model; 1-D array is interpreted as just one case, N=1
	GPA.resize(N,1)
	##Get all test scores
	#lm = linear_model.LinearRegression(fit_intercept = True)
	arr_rmse = np.zeros(5)
	for i in range(5):
		test = np.array(map(float,raw_input().strip().split()))
		print test
		test.resize(N,1)
		#test_mat.append(test)
		lm = linear_model.LinearRegression(fit_intercept = True)
		lm.fit(test, GPA)
		err = np.array(map(lm.predict, test))
		arr_rmse[i] = np.sqrt(np.sum(err * err)/N)
	arr_result[t] = arr_rmse.argmin() + 1
	print arr_rmse
	#test_mat = np.array(test_mat)
for result in arr_result:
	print result

