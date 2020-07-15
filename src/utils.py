import cv2
import matplotlib.pyplot as plt
from os import listdir, remove
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import log_loss


class ImageOperations():
    def __init__(self):
        self.availableExtensions = ['png', 'PNG', 'jpg', 'jpeg', 'bmp', 'tif']
        super()
    

    def showImage(self, path: str):
        # to show image
        img = cv2.imread(path)
        height, width, channels = img.shape
        print(str(height) + " " + str(width) + " " + str(channels))
        plt.imshow(img)
        plt.show()
    
    def countNumberOfImagesInFolder(self, folder: str):
        names = [name for name in listdir(folder) if self.isImageExtension(name)]
        return len(names)
    
    def isImageExtension(self, imageName: str):
        extension = imageName.split('.')[-1]
        if extension in self.availableExtensions:
            return True
        else:
            return False
        
    def removeImages(self, imagesFolder: str):
        for i in listdir(imagesFolder):
            extension = i.split('.')[-1]
            if not extension in self.availableExtensions:
                print('{} is removed'.format(i))
                remove(imagesFolder + i)




class MetricUtils():
    def __init__(self):
        super()
    
    def getROCValues(self, y_true, y_pred):
        return roc_curve(y_true, y_pred)

    def getLogLoss(self, y_true, y_pred):
        return log_loss(y_true, y_pred)



class GraphUtils():
    def __init__(self):
        super()

    # def frequencyPlot(xlabel: str, ylabel: str, title: str, data: dict):
    #     df = pd.Dataframe(data)
    #     df.hist()


        