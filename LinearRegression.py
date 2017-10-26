
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'cleaned_train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'cleaned_test.csv' #replace
ALPHA = 12e0
EPOCHS =75000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model1'
train_flag = True


logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #remove this line once you finish writing
	col=np.ones((X.shape[0],1))
	arr=np.hstack((col,X))
	arr=np.array(arr)
	return arr
	



 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
	arr1=np.zeros(n_thetas)
	return arr1



def train(theta, X, y, model):
	J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
    #train for the number of epochs you have defined
	m = len(y)
    #your  gradient descent code goes here
    #steps
    #run you gd loop for EPOCHS that you have defined
    #calculate the predicted y using your current value of theta
    # calculate cost with that current theta using the costFunc function
    #append the above cost in J
    #calculate your gradients values using calcGradients function
    # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
	for i in range(EPOCHS):
		a=predict(X,theta)
		pr=1.0/(1+np.exp(-a))
		error=costFunc(m,y,pr)
		J.append(error)
		c=calcGradients(X,y,pr,m)
		theta=makeGradientUpdate(theta,c)
	model['J'] = J
	model['theta'] = list(theta)
	return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
	total_sum = np.sum(y*np.log(y_predicted)+(1-y)*np.log(1-y_predicted))/(m)
	#print total_sum
	return total_sum

def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
	arr=((np.sum((np.subtract(y_predicted,y)).reshape((X.shape[0],1))*X,axis=0)))/m
	return arr
	
		
	
	
#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
	arr=np.subtract(theta,ALPHA*grads)
	return arr

#this function will take two paramets as the input
def predict(X,theta):
	arr2=np.dot(X,theta)
	return arr2


########################main function###########################################
def main():
	if(train_flag):
		model = {}
		X_df,y_df = loadData(FILE_NAME_TRAIN)
		X,y, model = normalizeData(X_df, y_df, model)
		X = appendIntercept(X)
		theta = initialGuess(X.shape[1])
		model = train(theta, X, y_df, model)
		with open(MODEL_FILE,'w') as f:
			f.write(json.dumps(model))
	if(train_flag):
		model = {}
		with open(MODEL_FILE,'r') as f:
			model = json.loads(f.read())
			X_df, y_df = loadData(FILE_NAME_TEST)
			X,y = normalizeTestData(X_df, y_df, model)
			X = appendIntercept(X)
			accuracy(X,y_df,model)

if __name__ == '__main__':
    main()
