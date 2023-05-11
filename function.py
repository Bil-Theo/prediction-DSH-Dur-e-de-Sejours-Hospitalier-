def initiation():
    return np.random.randn(2,1)

def modele(x, tetha):
    return x.dot(theta)

def function_cost(x, y, theta):
    m = len(y)
    return 1/(2*m)*(np.sum(modele(x, theta) -  y)**2)

def gradient(x, y, theta):
    m = len(y)
    return (1/m) * np.sum( x.T.dot(modele(x, theta) - y))

def gradient_Desc(x, y, theta, nbr_iterations, learning_rate):
    cost_History = np.zeros(nbr_iterations)

    for i in range(1, nbr_iterations):
        theta = theta - (learning_rate * gradient(x, y, theta))
        cost_History[i] = function_cost(x, y, theta)
    
def RMSE(y_rel, y_pred):
    u = ((y_rel - y_pred)**2).sum()
    v = ((y_rel - y_rel.mean())).sum()

    return u/v

def prediction(x, theta):
    return modele(x, theta)
