#import torch
from backpropagation import Value
import random


class Neuron:

    # nin : number of inputs for the neuron
    def __init__(self,nin,notrandom=False) -> None:
        if notrandom:
            self.w = []
            for i in range(nin):
                self.w.append(Value(0.11 + i*0.01))
                self.w[-1].label = f"w{i}"
            self.b = Value(0.22)
            self.b.label = "b"
        else:
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b = Value(random.uniform(-1, 1))
        self.nin = nin
        self.parameters = self.w + [self.b]

    # used for the syntax neuron(x)
    # forward pass 
    def __call__(self, x):
        # w*x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        act.label = "wixi"
        out = act.tanh()
        return out

class Layer:

    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.parameters = [p for neuron in self.neurons for p in neuron.parameters]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:

    # nin : number of inputs
    # nouts : list of nout
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.parameters = [p for layer in self.layers for p in layer.parameters]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        repr = f"Parameters : {len(self.parameters)}\n"
        repr += f"Layers : {len(self.layers)}\n"
        nb_neurons = 0
        for i,l in enumerate(self.layers):
            repr += f"Layer {i} : {len(l.neurons)}\n"
        
        return repr
def mlp_test():
    n = Neuron(2)
    print(n.w)
    print(n.b)
    # 3 neurons, that takes 2 parameters as input
    l = Layer(2,3)
    x = [1.0,2.0]
    print(l(x))
    # Sample MLP : 3 inputs, 
    # 3 layers : L1 has 4 neurons, L2 has 4 neurons, L3 (output) has 1 neuron 
    mlp = MLP(3, [4, 4, 1])
    x = [2.0, 3.0, -1.0]
    print("MLP")
    print(mlp(x))

def gradient_test():
    # sample x
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    # target values
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    #mlp = MLP(3, [4, 4, 1]) # 3 inputs, 2 hiddens layers of 4 neurons, 1 output
    mlp = MLP(3, [4, 4, 1])
    
    print(f"Parameters : {len(mlp.parameters)}")
    print(mlp)
    epoch =100
    for i in range(epoch):
        # compute predictions for all our samples
        ypred = [mlp(x) for x in xs]
        # compute loss
        # ygt : y ground truth
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        if i % 10 ==0:
            print(f"{i:3}:{loss.data}")
        loss.backward()
        lr = 0.01

        for p in mlp.parameters:
            p.data = p.data - lr*p.grad
            p.grad = 0

def neuron_test(slice=0):
    # sample x
    xs = [
        [2.7, 37, -1.7],
        [3.7, -1.7, 0.7],
        [0.7, 1.7, 1.7],
        [1.7, 1.7, -1.7],
    ]
    # target values
    ys = [1.9, -1.9, -1.9, 1.9] # desired targets
    xs =xs[slice:]
    ys =ys[slice:]
    print(xs)
    print(f"Value to predicts : {len(ys)}")
    #mlp = MLP(3, [4, 4, 1]) # 3 inputs, 2 hiddens layers of 4 neurons, 1 output
    n = Neuron(3,notrandom=True)
    for i,p in enumerate(n.parameters):
        print(p)

    epoch =2
    print_step = 1
    for i in range(epoch):
        # compute predictions for all our samples
        ypred = [n(x) for x in xs]
        
        for y in ypred:
            print("\n\nBEFORE YPRED ---")
            for node in y.get_topo_nodes():
                print(node)
            print("END YPRED --- \n\n")

        # compute loss
        # ygt : y ground truth
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        print("\n\nTOPO")
        for node in loss.get_topo_nodes():
            print(node)
        print("END TOPO\n\n")
        if i % print_step ==0:
            print(f"{i:3}:{loss.data}")
        loss.backward()
        print("\n\nAfter backward ---")
        for node in loss.get_topo_nodes():
            print(node)
        print("END TOPO --- \n\n")
        lr = 0.01

        for p in n.parameters:
            p.data = p.data - lr*p.grad
            p.grad = 0

    print(f"FINAL loss={loss}")




if __name__ == '__main__':
    
    neuron_test(2)
    #neuron_test()






