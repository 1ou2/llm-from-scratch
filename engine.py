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

    def __repr__(self) -> str:
        repr = f"Value={self.data} - ID={self.id} OP={self._op} GRAD={self.grad} -children="
        for c in self._prev:
            repr = repr + str(c.id) + " " 
        return repr

    # override addition 
    # V1 + V2 is the same V1.__add__(V2)
    def __add__(self,other):
        v = Value(self.data + other.data,(self,other),'+')
        return v
    
    def __mul__(self,other):
        v = Value(self.data*other.data,(self,other),'*')
        return v

a = Value(3.0)
b = Value(2.1)
print(a,b)
print(a+b)
print(a*b)
