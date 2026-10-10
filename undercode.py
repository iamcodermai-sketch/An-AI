from collections.abc import Sized
from random import uniform, randint
import matplotlib.pyplot as plt
import math

def rl(x: Sized):
    return range(len(x))

def sig(x: float):
    return 1/(1+math.exp(-x))

class AI:
    class NEURON:
        def __init__(self, bias: float | None = None):
            self.value: float = 0
            self.bias: float = bias or 0

    class CONNECTION:
        def __init__(self, weight: float | None = None, neurons: tuple[int, int] | None = None):
            self.neurons: tuple[int, int] = neurons or []
            self.weight: float = weight or 0

    def __init__(self, layers):
        self.func = math.sin
        self.effeciency = []
        self._oUIDs: list[int] = []
        self._neus: list[AI.NEURON] = []
        self._cons: list[AI.CONNECTION] = []
        self.dataset: list[list[float]] = [[math.pi * (i-500) / 1000] for i in range(2000)]

        for layerUID in rl(layers):
            for row in range(layers[layerUID]):
                self._neus.append(AI.NEURON(uniform(-1, 1)))
                if layerUID == len(layers)-1:
                    self._oUIDs.append(len(self._neus)-1)

        for layerUID in rl(layers)[:-1]:
            for fromneuID in range(layers[layerUID]):
                for toneuID in range(layers[layerUID+1]):
                    self._cons.append(AI.CONNECTION(uniform(-0.01, 0.01), (fromneuID+sum(layers[:layerUID]), toneuID+sum(layers[:layerUID+1]))))

    def ask(self, data: list[float]):
        self.prepare_ask(data)
        return self.solve()

    def prepare_ask(self, data: list[float]):
        i = 0
        for n in self._neus:
            n.value = n.bias
        for j in data:
            self._neus[i].value = sig(j)
            i+=1

    def solve(self):
        for conn in self._cons:
            if self._neus[conn.neurons[0]].value > self._neus[conn.neurons[1]].value:
                f = conn.neurons[0]
                t = conn.neurons[1]
            else:
                f = conn.neurons[1]
                t = conn.neurons[0]

            self._neus[t].value += sig(self._neus[f].value) * conn.weight
        return [math.floor(self._neus[i].value*100)/100 for i in self._oUIDs]

    def c_mse(self):
        errs = []
        for datai in rl(self.dataset):
            if datai > 0:
                errs.append((-(self.func(self.dataset[datai-1][0]) - self.func(self.dataset[datai][0])) - self.ask(self.dataset[datai])[0])**2)
            else:
                errs.append((-(0 - self.func(self.dataset[datai][0])) - self.ask(self.dataset[datai])[0])**2)
        self.mse = sum(errs)
        self.effeciency.append(self.mse)
        if len(self.effeciency) > 100:
            self.effeciency = self.effeciency[len(self.effeciency)-100:]
        return self.mse

    def show(self):
        values: list[float] = []
        for i in self.dataset:
            if len(values):
                values.append(self.ask(i)[0] + values[len(values)-1])
            else:
                values.append(self.ask(i)[0])
        self.ax[0, 0].cla()
        self.ax[0, 0].set_title('MSE')
        self.ax[0, 0].plot(rl(self.effeciency), self.effeciency, color='#00ff00')
        self.ax[0, 1].cla()
        self.ax[0, 1].set_title('Predict')
        self.ax[0, 1].plot(rl(values), values, color='#0000ff')
        rvalues: list[float] = [self.func(i[0]) for i in self.dataset]
        self.ax[1, 0].cla()
        self.ax[1, 0].set_title('Original')
        self.ax[1, 0].plot(rl(rvalues), rvalues, color='#ff0000')

        plt.show(block=0)
        plt.pause(0.0001)

    def teach(self):
        self._old_mse = self.c_mse()
        wigb = randint(0, 1)
        if wigb == 0:
            target = randint(0, len(self._neus)-1)
            delta = uniform(-1, 1)

            self._neus[target].bias += delta
        else:
            target = randint(0, len(self._cons)-1)
            delta = uniform(-1, 1)

            self._cons[target].weight += delta
        if self.c_mse() > self._old_mse:
            if wigb == 0:
                self._neus[target].bias -= delta
            else:
                self._cons[target].weight -= delta
            self.c_mse()
        self.show()

    def teach_cycle(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 2)
        while 1:
            self.teach()