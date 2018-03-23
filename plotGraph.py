import matplotlib.pyplot as plt

class PlotGraph(object):
    
    def plot(self,nb_epochs,loss,type):
        x = range(1,nb_epochs+1)
        plt.plot(x,loss)
        plt.xlabel("Number of epochs")
        plt.ylabel("Training loss")
        title = type+" Loss with increasing epochs"
        plt.title(title)
        plt.show()
    
