import numpy as np
import scipy.signal as sp

class ConvolutionalNeuralNetwork:
    def __init__(self, input:np.ndarray):
        self.input = input
    def run(self):
        Kernel1 = Kernel(np.random.uniform(-1,1,(3,3)), 1)
        PoolingLayer1 = PoolingLayer(2, 2)
        Kernel2 = Kernel(np.random.uniform(-1,1,(3,3,10)), 1)
        PoolingLayer2 = PoolingLayer(3, 1)
        Kernel3 = Kernel(np.random.uniform(-1,1,(3,3,6)), 1)
        PoolingLayer3 = PoolingLayer(3, 1)
        for image in self.input:
            currentLayer = Layer(image)
            currentLayer = Kernel1.apply(currentLayer)
            currentLayer = PoolingLayer1.apply(currentLayer)
            currentLayer = Kernel2.apply(currentLayer)
            currentLayer = PoolingLayer2.apply(currentLayer)
            currentLayer = Kernel3.apply(currentLayer)
            currentLayer = PoolingLayer3.apply(currentLayer)

class Layer:
    def __init__(self, matrix:np.ndarray) -> None:
        self.matrix:np.ndarray = matrix
        self.W = matrix[0].size   
    def addPadding(self) -> None:
        newMatrix:np.ndarray = np.zeros((self.W+2, self.W+2))
        newMatrix[1:self.W+1, 1:self.W+1] = self.matrix
        self.W += 2
        self.matrix = newMatrix
        
class Kernel:
    def __init__(self, filter:np.ndarray, stride:int) -> None:
        self.filter = filter
        self.stride = stride
        self.F = filter[0].size  
    def apply(self, layer:Layer) -> Layer:
        return Layer(sp.convolve(layer.matrix, self.filter[::-1, ::-1], mode='valid')[::self.stride, ::self.stride])
    def setStride(self, newStride: int) -> None:
        self.stride = newStride
    def setFilter(self, newFilter: np.ndarray) -> None:
        self.filter = newFilter
  
# Max Pooling  
class PoolingLayer:
    def __init__(self, spatialSize:int, stride:int) -> None:
        self.spatialSize = spatialSize
        self.stride = stride
    def apply(self, layer:Layer) -> Layer:
        return Layer(sp.convolve(layer.matrix, self.filter[::-1, ::-1], mode='valid')[::self.stride, ::self.stride])
    def setStride(self, newStride: int) -> None:
        self.stride = newStride
    
class FlatteningLayer:
    def __init__(self, layer: Layer) -> Layer:
        return Layer(layer.matrix.flatten())
    
class RELU:
    def __init__(self, layer: Layer) -> Layer:
        return Layer(layer.matrix * (layer.matrix * 0))
    
class FullyConnectedLayer:
    def __init__(self, layer: Layer):
        self.layer = layer 
        self.weights = np.random.uniform(-1,1,layer.matrix[0].size)
    def h(self) -> np.ndarray:
        return np.matmul(self.layer.matrix, self.weights) 
    def loss_function(self, x, y, theta): # CHECK #
	    return ((self.h()-y).T@(self.h()-y))/(2*y.shape[0])
        
        
        
        
        
"""
def JICapply(self, layer:Layer) -> np.ndarray:
        activationMap:np.ndarray = np.zeros((int((layer.W - self.F)/self.stride)+1, int((layer.W - self.F)/self.stride)+1))
        mapX, mapY, imageX, imageY = 0, 0, 0, 0
        while(imageX+self.F <= layer.W):
            while(imageY+self.F <= layer.W):
                scanning = layer.matrix[imageX:imageX+self.F, imageY:imageY+self.F]
                scanning = scanning * self.filter
                activationMap[mapX, mapY] = scanning.sum()
                mapY += 1
                imageY += self.stride
            imageY, mapY = 0, 0
            mapX += 1
            imageX += self.stride
        return activationMap
"""