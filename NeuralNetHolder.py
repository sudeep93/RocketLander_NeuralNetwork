# import pandas as pd
import random
import math
from math import sqrt
from random import randrange
# load anything you need, initialize anything you need.
def deNormaliseDataSet(normaliseData):
    normVal1 = normaliseData[0][0] - (-7.927907828)/(7.995825017-(-7.927907828))
    normVal2 = normaliseData[1][0] - (-5.734376934)/(5.95205551-(-5.734376934))
    deNormalisedData=[normVal1,normVal2]
    return deNormalisedData

def sigmoid(x):
    #applying the sigmoid function
    return 1 / (1 + math.exp(-(x)))


def derSigmoid(self,y):
        #applying the dervative sigmoid function
    return y*(1-y)




def MA(X,Y):
    return [[X[i][j] + Y[i][j]  for j in range(len(X[0]))] for i in range(len(X))]

def MS(X,Y):
    return [[X[i][j] - Y[i][j]  for j in range(len(X[0]))] for i in range(len(X))]

def MT(X):
    return  [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

def MM(X,Y):          
    return [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

def createMatrixRandom(rowCount, colCount):
    mat = []
    for i in range(rowCount):
        rowList = []
        for j in range(colCount):
            rowList.append(round(random.uniform(0.0,1.0), 10))
        mat.append(rowList)
    return mat

def createMatrixofOne(rowCount, colCount):
    mat = []
    for i in range(rowCount):
        rowList = []
        for j in range(colCount):
        # you need to increment through dataList here, like this:
            rowList.append(1)
        mat.append(rowList)
    return mat


def createMatrixofInputorOutput(datalist):
    k=0
    for i in range(1):
        rowList = []
        for j in range(len(datalist)):
            rowList.append([datalist[k]])
            k=k+1
    return rowList


def func_normalize(input):
    max_val=max(input)
    min_val=min(input)
    output=[]
    for inputs in input:
        output.append((inputs-min_val)/(max_val-min_val))
    return output

# def map_sigmoid(matrix):
#     rows = len(matrix)
#     cols = len(matrix[0])
#     for i in range(rows):
#         for j in range(cols):
#             matrix[i][j]=1 / (1 + (math.exp(-matrix[i][j])))
#     return matrix

def map_sigmoid(matrix,lamda):
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] < 0.0:
                matrix[i][j]= 1 - 1 / (1 + math.exp(lamda*matrix[i][j]))
            else:
                matrix[i][j]=1 / (1 + (math.exp(-lamda*matrix[i][j])))
    return matrix

def multiply(matrix,n):
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            matrix[i][j]*=n
    return matrix

def multiply_hm(matrix,n):
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            matrix[i][j]*=n[i][j]
    return matrix


def addition(matrix,n):
#         print(matrix)
    rows = len(matrix)
    cols = len(matrix[0])
#         print(rows,cols)
    for i in range(rows):
        for j in range(cols):
            matrix[i][j]+=n

    return matrix

def addition_hm(matrix,n):
#         print(matrix)
    rows = len(matrix)
    cols = len(matrix[0])
#         print(rows,cols)
    for i in range(rows):
        for j in range(cols):
            matrix[i][j]+=n[i][j]

    return matrix


def map_derSigmoid(matrix):
#         print(matrix)
    rows = len(matrix)
    cols = len(matrix[0])
#         print(rows,cols)
    for i in range(rows):
        for j in range(cols):
            matrix[i][j]=matrix[i][j]*(1-matrix[i][j])               
    return matrix



# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.input_nodes      = 2
        self.hidden_nodes      = 3
        self.output_nodes   = 2
        self.weight_inh = [[5.498044615730283e-05, -5.652008405520862e-05], [0.7609779913703038, 0.784658917532949], [0.8498774544495037, 0.7959245107225534]]#createMatrixRandom(self.hidden_nodes,self.input_nodes)
        self.weight_hop = [[0.8318024229772629, -0.4482663281047039, -0.4434222709290466], [0.19068997026892226, 0.2304742912458903, 0.1391549882898376]]#createMatrixRandom(self.output_nodes,self.hidden_nodes) 
        self.bias_inh=[[0.01787418205194673], [13.181769464314065], [13.59564123082825]]#createMatrixRandom(self.hidden_nodes,1)#[[0.1],[0.6]]#createMatrixRandom(self.hidden_nodes,1)
        self.bias_hop=[[0.7216620267618108], [-0.3848193598282009]]#[[1.6]]#createMatrixRandom(self.output_nodes,1)
        self.learning_rate=0.87  #learning_rate
        self.lamda=0.6#lamda
     

    
    def predict(self, input_row):
        input_rows=input_row.split(",")
        inputs=[]
        inputs.append([float(input_rows[0])-(-643.3285662)/1260.405928])
        inputs.append([float(input_rows[1])-(65.13639224)/872.2587613])
        print("---------->",inputs)
        hid_val=MM(self.weight_inh,inputs)# matrix multi
        hid_bias=MA(hid_val,self.bias_inh)#matrix addition
        hid_act_func=map_sigmoid(hid_bias,self.lamda)
        out_val=MM(self.weight_hop,hid_act_func)
        out_bias=MA(out_val,self.bias_hop)
        output=map_sigmoid(out_bias,self.lamda)
        outputs=deNormaliseDataSet(output)
        return outputs
    

    
        
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        
		#Input_row is string like: x_distance,y_distance
		#	Split string, normalize values, pass them through the feedforward process and denormalize the results
		
		#return a list like --> [X_Velocity, Y_Velocity]


