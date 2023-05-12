import matplotlib.pyplot as plt

class graphic:
    
    def __init__(self, dimension):
        self.dimension = dimension
        self.fig = plt.figure(figsize=dimension)
        self.ax = self.fig.add_subplot(111)

    def graph2D(self, x, y, y_pred, label = 'graph', color = 'blue'):
        self.ax.scatter(x[:, 20], y, c = color)
        self.ax.scatter(x[:, 20], y_pred, color='red')
        plt.title(label)
        plt.savefig('./images/fig_datasets_In_2D.png')
        plt.show()
    
    def graphCost(tab, label = 'graph', color = 'blue'):
        plt.plot(range(tab.shape[0]), tab)
        plt.title(label)
        plt.savefig('./images/fig_cost_evolution.png')
        plt.show()
    
    def graphRMSE(self, tab, label = 'graph', color = 'red'):
        self.ax.plot(range(tab.shape[0]), tab)
        plt.title(label)
        plt.savefig('./images/fig_RMSE_evolution.png')
        plt.show()
