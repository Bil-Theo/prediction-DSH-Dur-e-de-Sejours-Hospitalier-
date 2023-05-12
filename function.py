import numpy as np
import pickle
import graph

def initiation(x):
    return np.random.randn(x.shape[1],1)

def modele(x, theta):
    return x.dot(theta)

def function_cost(x, y, theta):
    m = len(y)
    return 1/(2*m)*(np.sum(modele(x, theta) -  y)**2)

def gradient(x, y, theta):
    m = len(y)
    return (1/m) *  x.T.dot(modele(x, theta) - y)

def gradient_Desc(x, y, nbr_iterations, learning_rate, goal):
    cost_History = np.zeros(nbr_iterations)
    rmse = np.zeros(nbr_iterations)
    theta =  initiation(x)
    for i in range(0, nbr_iterations):
        theta = theta - learning_rate * gradient(x, y, theta)
        cost_History[i] = function_cost(x, y, theta)
        y_pred = prediction(x, theta)
        rmse[i] = 1 - RMSE(y, y_pred)

        if((1 - rmse[i]) >= goal):
            break
    
    return theta, cost_History, rmse
    
def RMSE(y_rel, y_pred):
    u = ((y_rel - y_pred)**2).sum()
    v = ((y_rel - y_rel.mean())**2).sum()

    return 1 - u/v

def prediction(x, theta):
    return modele(x, theta)


def save_model(theta):
    with open('./datasets/modele.data', 'wb') as file:
        myPickel = pickle.Pickler(file)
        myPickel.dump(theta)
    
    file.close()

def load_modele():
    with open('./datasets/modele.data', 'rb') as file:
        myPickel = pickle.Unpickler(file)
        theta = myPickel.load()
    file.close()
    
    return theta

def train(x, y, nbr_iterations = 1000, learning_rate = 0.01, goal = 0.98, save_modele = False, visual_result =  False):

    while(True):
        theta, cost_History, rmse = gradient_Desc(x, y, nbr_iterations, learning_rate, goal)
        y_predite = prediction(x, theta)
        a = RMSE( y, y_predite)
        print("Performance: ",a)
        if( a>= goal):
            print("Meilleur Performance: ",a)
            break
    
    if(save_modele):
        save_model(theta)
    
    if(visual_result):
        myGraph = graph.graphic((7,7))
        myGraph.graph2D(x, y, color = 'blue')
        myGraph.graphCost(cost_History)
        myGraph.graphRMSE(rmse)