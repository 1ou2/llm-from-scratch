
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# Gradient tutorial
#
# a value in the matrix
class Value:
    # global ID - used for debug and priting
    gid = 0

    def __init__(self,data:float,_children=(),_operation='') -> None:
        # a float representing the value
        self.data = data
        # gradient
        # represents the derivative of the end Value (the loss function) with respect to this current Value objet
        self.grad = 0.0
        # the predecessors that were used to compute this value
        self._prev = set(_children)
        # the operation used ("*" for multiplication or "+" for addition)
        self._op = _operation
        # id of this node
        self.id = Value.gid
        # increment global counter
        Value.gid +=1

    # for print purpose
    def __repr__(self) -> str:
        repr = f"Value={self.data} - ID={self.id} OP={self._op} GRAD={self.grad} -children="
        for c in self._prev:
            repr = repr + str(c.id) + " " 
        return repr

    # override addition 
    # V1 + V2 is the same V1.__add__(V2)
    # returns a new Value object
    def __add__(self,other):
        # data value obtained by adding the two floats
        # the children of this new value object are the current Value object and other
        # operation is "+"
        v = Value(self.data + other.data,(self,other),'+')
        # TODO : check if self == other
        v.grad += other.grad
        return v
    
    def __mul__(self,other):
        v = Value(self.data*other.data,(self,other),'*')
        v.grad = self.grad * other.data + self.data * other.grad
        return v
    
    def backward(self):
        self.grad = 1.0
        visited = list()

        def build_topo(v):
            if v not in visited:
    
                for child in v._prev:
                    build_topo(child)
                # only append if all children are visited
                visited.append(v)    

        build_topo(self)
        for node in reversed(visited):
            #node.grad += node.grad * node.data
            print(node)


a = Value(3.0)
b = Value(2.1)
c = Value(10.0)
d = a*b + c
e = a + b
f = d*e

f.backward()
#print(a,b)
#print(a+b)
#print(a*b)

#print(d)
ddot = draw_dot(f)
ddot.render()
#ddot
#ddot.render('gout')