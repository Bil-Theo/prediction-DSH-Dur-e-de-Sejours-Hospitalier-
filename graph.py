import matplotlib.pyplot as plt

class graphic:
    
    def __init__(self, dimension):
        self.dimension = dimension
        self.fig = plt.figure(figsize=dimension)
        self.ax = self.fig.add_subplot(111)

    def graph2D(self, x, y, label = 'graph', color = 'blue'):
        self.ax.scatter(x[:, 0], y, c = color)
        plt.title(label)
        plt.savefig('./images/fig_datasets_In_2D.png')
        plt.show()
    
    def graphCost(self, tab, label = 'graph', color = 'blue'):
        self.ax.plot(range(tab.shape[0]), tab)
        plt.title(label)
        plt.savefig('./images/fig_cost_evolution.png')
        plt.show()
    
    def graphRMSE(self, tab, label = 'graph', color = 'red'):
        self.ax.plot(range(tab.shape[0]), tab)
        plt.title(label)
        plt.savefig('./images/fig_RMSE_evolution.png')
        plt.show()
