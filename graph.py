import matplotlib.pyplot as plt

class graphic:
    
    def __init__(self, title, dimension):
        this.title = title
        this.dimension = dimension
        this.fig = plt.figure(figsize=dimension)
        this.ax = this.fig.add_subplot(111)

    def graph2D(x, y, color):
        this.ax.scatter(x[:, 0], y, c = color)
        plt.savefig('./images/fig_datasets_In_2D.png')
        plt.show()
