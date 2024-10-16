import math
import torch

def torch_backpropagation():
    x1 = torch.Tensor([2.0]).double() ; x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double() ; x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double() ; w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double() ; w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double() ; b.requires_grad = True

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    o = torch.tanh(n)
    print(o.data.item())
    o.backward()

    for i in [x1, x2, w1, w2, b]:
        print(i.grad.item())

class Value:
    # global counter for all Value objects
    _id = 0

    @staticmethod
    def _get_id():
        Value._id += 1
        return Value._id

    def __init__(self, data,prev=(),label=None):
        """ 
        data : float number representing the value
        prev : list of Value objects that this Value depends on
        label : string label for the Value object
        grad : float gradient of the Value object
        _backward : function to compute the gradient of the Value object
        id : unique identifier for the Value object
        """
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = prev
        self.id = Value._get_id()
        self.label = label

    def __call__(self, *args, **kwargs):
        return self.data
    
    def __repr__(self) -> str:
        prev = ""
        for p in self._prev:
            prev += f"{p.id},"
        return f"V[{self.id:2}|{self.label}|data={self.data}|{prev=}|grad={self.grad}]"
    
    def __add__(self,other):
        # check if other is a Value object
        if not isinstance(other, Value):
            # assume other is a numeric value
            other = Value(float(other))
        data = self.data + other.data
        out = Value(data,(self,other))

        # in case of addition, the gradient flows unchanged from parent to child
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other


    def __mul__(self,other):
        if not isinstance(other, Value):
            other = Value(float(other))
        data = self.data * other.data
        out = Value(data,(self,other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        data = self.data ** other
        out = Value(data, (self, ))

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
            

        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    # get all nodes in topological order
    def get_topo_nodes(self):
        seen = set()
        topo = []

        def build_topo(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        return topo

    def backward(self):
        # the derivative of the cost function with respect to the output is 1
        self.grad = 1
        # call _backward on all nodes
        for node in reversed(self.get_topo_nodes()):
            #print(node)
            node._backward()

def back_test1():
    a = Value(3.0,label="a")
    b = Value(2.0,label="b")
    c = a * b; c.label = "c"
    d = c -5; d.label = "d"
    e = d.tanh(); e.label = "e"
    e.backward()

    print("\n")
    for v in [a,b,c,d,e]:
        print(v)

def kaperthy_test():       
    w2 = Value(1.0, label="w2")
    x2 = Value(0.0, label="x2")
    x1 = Value(2.0, label="x1")
    w1 = Value(-3.0, label="w1")
    b = Value(6.8813735870195432, label="b")
    x2w2 = x2 * w2; x2w2.label = "x2w2"
    x1w1 = x1 * w1; x1w1.label = "x1w1"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"
    n = x1w1x2w2 +b; n.label = "n"
    o = n.tanh(); o.label = "o"
    o.backward()

    print("\n")
    for v in reversed([w2,x2,x1,w1,x2w2,x1w1,x1w1x2w2,n,o]):
        print(v)

def forward_test():
    # init w1,w2,b
    w2 = Value(0,5, label="w2")
    w1 = Value(-0.03, label="w1")
    b = Value(0.88, label="b")

    xs = [
        [1.0,0.0],
        [-1.5,0.7],
        [-0.2,-0.4]
    ]

    ys = [1.0,-1.0,-1.0]

    # predicted y
    ypred = []
    # forward pass
    for sample in xs:
        x1 = Value(sample[0],label="x1")
        x2 = Value(sample[0],label="x2")
        x2w2 = x2 * w2; x2w2.label = "x2w2"
        x1w1 = x1 * w1; x1w1.label = "x1w1"
        x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"
        n = x1w1x2w2 +b; n.label = "n"
        o = n.tanh(); o.label = "o"
        ypred.append(o)

    #loss = Value(0.0,label="loss")
    #for ygt, yout in zip(ys,ypred):
    #    ygt = Value(ygt)
    #    loss += (yout-ygt)**2

    loss = sum( (yout-ygt)**2 for ygt, yout in zip(ys,ypred))
    print(loss)
    loss.backward()

if __name__ == "__main__":
    #back_test1()
    #kaperthy_test()
    forward_test()