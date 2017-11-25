
# coding: utf-8

# In[1]:

import numpy as np


# **Module** - это абстрактный класс определяющий необходимые для нейронной сети методы. В этой части не нужно ничего писать, просто читайте комментарии.

# In[2]:

class Module(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    """
    В целом, Module это черный ящик, который принимает что-то на вход и
    выдает какой-то результат. Это результат применения метода "forward":
        
        output = module.forward(input)
    
    Также этот черный ящик должен уметь по значеню dL/dOutput подсчитывать
    производную функции ошибки по параметрам модели (dL/dW) и по входным данным
    (dL/dInput)
    
        gradInput = module.backward(input, gradOutput)
    """
    
    def forward(self, input):
        """
        Вычислает по данному input соответствующий output, сохраняя его в одноименном
        поле себя.
        """
        
        # Пример для тождественной функции: 
        
        # self.output = input
        # return self.output
        
        assert False, 'Implementation error'

    def backward(self, input, gradOutput):
        """
        Делает шаг обратного распространения ошибки.
        
        Включает в себя:
        1) подсчет градиента функции ошибки по входу (для выполнения
           шага обратного распространения ошибки предыдущего слоя)
        2) подсчет градиента функции ошибки по параметрам (для 
           выполнения шага градиентного спуска)
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        """
        Подсчитывает градиент функции ошибки по входу, сохраняет результат
        в поле gradInput. Размерность полученного градиента всегда строго совпадает
        с размерностью входа.
        """
        
        # Для тождественной функции:
        
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        assert False, 'Implementation error'
    
    def accGradParameters(self, input, gradOutput):
        """
        Подсчитывает градиент функции потерь по параметрам
        
        Не нужно реализовывать если у модуля нет параметров (ReLU)
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Обнуляет градиент
        """
        pass
        
    def getParameters(self):
        """
        Возвращает список подгоняемых (trainable) параметров.
        
        Если таковых нет, то возвращает пустой список.
        """
        return []
        
    def getGradParameters(self):
        """
        Возвращает список градиентов функции потерь по подгоняемым параметрам.
        
        Если таковых нет, то возвращает пустой список.
        """
        return []
    
    def train(self):
        """
        Переводит модуль в режим обучение. Влияет только на дропаут и batchNorm.
        """
        self.training = True
    
    def eval(self):
        """
        Отключает режим обучения. Влияет только на дропаут и batchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Выводит имя модуля.
        """
        return "Module"


# # Sequential container

# **Define** a forward and backward pass procedures.

# In[3]:

class Sequential(Module):
    """
         Это контейнер, который для данного входа последовательно применяет
         все переданные в него модули (выход предущего модуля - вход следующего)
         
         Итоговое значение будет являться выходом данного контейнера.
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        """
        Добавляет модуль в контейнер.
        """
        self.modules.append(module)

    def forward(self, input):
        """
        Вычислает по данному input соответствующий output, сохраняя его в одноименном
        поле себя.
        
            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})
        """

        self.inputs = []
        last_output = input
        
        for i, mod in enumerate(self.modules):
            self.inputs.append(last_output)
            last_output = mod.forward(self.inputs[-1])
            
        self.output = last_output
                
        return self.output

    def backward(self, input, gradOutput):
        """
        Реализует backward pass:
            
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   
        
        """
        yinp = self.m_inputs[::-1]
        t = gradOutput
        
        for i, mod in enumerate(self.modules[::-1]):
            t = mod.backward(yinp[i], t)
                
        self.gradInput = t
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Возвращает список списков параметров каждого модуля.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Возвращает список списков градиентов функции потерь по параметрам каждого модуля.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)


# # Layers

# - input:   **`batch_size x n_feats1`**
# - output: **`batch_size x n_feats2`**

# In[4]:

class Linear(Module):
    """
    Модуль выполняющий линейное преобразование над входом. Также известен
    как полносвязный слой.
    
    Модуль должен принимать на вход матрицу размерности (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        
        self.gradW = np.zeros_like(self.W)
        
    def dumb_forward(self, input):
        '''
        Считает выход линейного слоя по данному входу, сохраняет результат в self.output
        и вовзращает self.output.
        output[object_idx, out_neuron_idx] = 
        \sum_{in_neuron_idx} input[object_idx,n] * w_[out_neuron_idx, in_neuron_idx]
        '''
        n_obj = len(input)
        n_out, n_in = self.W.shape
        self.output = np.zeros((n_obj, n_out))
        
        for i in range(n_obj):
            for j in range(n_out):
                for k in range(n_in):
                    self.output[i][j] += self.W[j][k] * input[i][k]
        
        return self.output
    
    def forward(self, input):
        self.output = np.dot(self.W, input.T).T
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        assert False, "Implementation error"

    def accGradParameters(self, input, gradOutput):
        assert False, "Implementation error"
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W]
    
    def getGradParameters(self):
        return [self.gradW]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q


# This one is probably the hardest but as others only takes 5 lines of code in total. 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**

# # Функции активации

# ReLU **Rectified Linear Unit** реализован за вас: 

# In[5]:

class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"


# In[6]:

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def forward(self, input):
        self.output = 1/(1 + np.exp(-input))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = self.output * (1 - self.output)
        return self.gradInput
    
    def __repr__(self):
        return "Sigmoid"


# # Criterions

# Функции потерь. 

# In[2]:

class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
            Считает ошибку, сохраняет ее и возвращает.
        """
        assert False, "Implementation error"

    def backward(self, input, target):
        """
            Считает градиент ошибки по input, сохраняет его в gradInput
            и возвращает.
        """
        self.updateGradInput(input, target)
        return self.gradInput

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        assert False, "Implementation error"

    def __repr__(self):
        """
        Выводит название функции потерь в человекочитаемом виде.
        """
        return "Criterion"


# The **MSECriterion**, which is basic L2 norm usually used for regression, is implemented here for you.

# In[13]:

class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"

