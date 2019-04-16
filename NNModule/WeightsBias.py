# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
from enum import Enum

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2

class WeightsBias(object):
    def __init__(self, n_input, n_output, eta, init_method = InitialMethod.Xavier):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.eta = eta

    def __GenerateWeightsArrayFileName(self):
        self.w_filename = str.format("w1_{0}_{1}_{2}.npy", self.num_output, self.num_input, self.init_method.name)

    def InitializeWeights(self, create_new = False):
        self.__GenerateWeightsArrayFileName()
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def __CreateNew(self):
        self.W, self.B = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        np.save(self.w_filename, self.W)
        
    def __LoadExistingParameters(self):
        w_file = Path(self.w_filename)
        if w_file.exists():
            self.W = np.load(w_file)
            self.B = np.zeros((self.num_output, 1))
        else:
            self.__CreateNew()
        # end if

    def Update(self):
        self.W = self.W - self.eta * self.dW
        self.B = self.B - self.eta * self.dB

    def toString(self):
        info = str.format("w={0}\nb={1}\n", self.W, self.B)
        return info

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.Zero:
            # zero
            W = np.zeros((num_output, num_input))
        elif method == InitialMethod.Normal:
            # normalize
            W = np.random.normal(size=(num_output, num_input))
        elif method == InitialMethod.Xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_output,num_input))
        # end if
        B = np.zeros((num_output, 1))
        return W, B
