import torch
import torch.nn as nn


def manualprop():
    # 1 Neuron
    w1 = torch.tensor([-1.0, 1.0])
    b = torch.tensor(1.0)

    # 1 sample from training data
    x1 = torch.tensor([1.0, 2.0])
    y = torch.tensor([0.8])


    b.requires_grad_(True)
    w1.requires_grad_(True)
    # retain_grad is used because we want to access gradient from intermediate nodes
    n1 = x1@w1 + b;n1.retain_grad()
    # use tanh as activation function
    o1 = torch.tanh(n1);o1.retain_grad()
    print(f"o1={o1}")

    # Mean square error between predicted output and real value
    loss = (y[0]-o1)**2 
    # start backpropagation using pytorch
    loss.backward()

    # First manually compute the influence of the bias b on the Loss
    # dL/db
    #
    print(f"dL/do1")
    print(f"{o1.grad.item()=}") 
    # L = (o1 -y0)² = o1² + y0² -2*o1*y0
    # dL/do1 = 2*o1 + 0 -2*y0 
    fo1 = 2*o1 - 2*y[0]
    print(f"{fo1.item()=}\n")

    print(f"do1/db")
    # g(x): x->tanh(x) // g'(x) = 1 - tanh(x)*tanh(x)
    # o1 = tanh(n1)
    fo1b = 1- torch.tanh(n1)**2
    print(f"{fo1b=}")

    print(f"dL/db")
    print(f"{b.grad=}")
    # dL/db = dL/o1 * do1/db
    fb = fo1*fo1b
    print(f"{fb=}")
    print(f"Computed gradient: {fb.item()=} vs pytorch gradient {b.grad.item()=}\n")

    #
    print(f"dL/dw11")
    print(f"{w1.grad=}")
    # dL/dw1 = dL/do1 * do1/dw1
    # dL/do1 = 2*o1- 2*y[0]
    # do1/dw1 = d(tanh(x1*w1+b))/dw1
    # g(x) = tanh(ax+b) // g'(x) = a * tanh'(ax+b) = a * (1 - tanh(ax+b)*tanh(ax+b))
    # a -> x1 = x1[0]
    fw11 = (2*o1- 2*y[0])*x1[0]*(1-torch.tanh(n1)**2)
    print(f"{fw11=}")

    print(f"dL/dw12")
    fw12 = (2*o1- 2*y[0])*x1[1]*(1-torch.tanh(n1)**2)
    print(f"{fw12=}")


def manualprop2():
    # 2 neurons same bias
    w1 = torch.tensor([-1.0, 1.0])
    w2 = torch.tensor([0.6, -0.7])
    b = torch.tensor(1.0)

    # 2 samples
    x1 = torch.tensor([1.0, 2.0])
    x2 = torch.tensor([4.0, 3.0])
    y = torch.tensor([0.8, 0.1])

    b.requires_grad_(True)
    w1.requires_grad_(True)
    w2.requires_grad_(True)

    n1 = x1@w1 + b;n1.retain_grad()
    o1 = torch.tanh(n1);o1.retain_grad()
    print(f"o1={o1}")
    n2 = x2@w2 + b;n2.retain_grad()
    o2 = torch.tanh(n2);o2.retain_grad()
    print(f"o2={o2}")

    loss = (y[0]-o1)**2 + (y[1]-o2)**2
    loss.backward()
    print(f"dL/do1")
    print(f"{o1.grad=}") 
    fo1 = 2*o1 - 2*y[0]
    print(f"{fo1=}")

    print(f"dL/do2")
    print(f"{o2.grad=}")
    fo2 = 2*o2 - 2*y[1]
    print(f"{fo2=}")


    print(f"do1/db")
    fo1b = 1- torch.tanh(n1)**2
    print(f"{fo1b=}")

    print(f"do2/db")
    fo2b = 1- torch.tanh(n2)**2
    print(f"{fo2b=}")

    print(f"dL/db")
    print(f"{b.grad=}")
    # dL/db = dL/o1 * do1/db + dL/o2 * do2/db
    fb = fo1*fo1b + fo2*fo2b
    print(f"{fb=}")



    print(f"dL/dw11")
    print(f"{w1.grad=}")
    # dL/dw1 = dL/do1 * do1/dw1
    # dL/do1 = 2*o1- 2*y[0]
    # do1/dw1 = d(tanh(x1*w1+b))/dw1
    # g(x) = tanh(ax+b) // g'(x) = a * tanh'(ax+b) = a * (1 - tanh(ax+b)*tanh(ax+b))
    # a -> x1 = x1[0]
    fw11 = (2*o1- 2*y[0])*x1[0]*(1-torch.tanh(n1)**2)
    print(f"{fw11=}")

    print(f"dL/dw12")
    fw12 = (2*o1- 2*y[0])*x1[1]*(1-torch.tanh(n1)**2)
    print(f"{fw12=}")
    print("\n")

    print(f"dL/dw21")
    print(f"{w2.grad=}")
    fw21 = (2*o2- 2*y[1])*x2[0]*(1-torch.tanh(n2)**2)
    print(f"{fw21=}")

    print(f"dL/dw22")
    fw22 = (2*o2- 2*y[1])*x2[1]*(1-torch.tanh(n2)**2)
    print(f"{fw22=}")


def manualprop3():
    
    # 1 Neuron
    w1 = torch.tensor([-1.0, 1.0])
    b = torch.tensor(1.0)
    # training data : 2 samples
    x1 = torch.tensor([1.0, 2.0])
    x2 = torch.tensor([4.0, 3.0])
    y = torch.tensor([0.8, 0.1])

    b.requires_grad_(True)
    w1.requires_grad_(True)
    
    # forward pass with x1
    n1 = x1@w1 + b;n1.retain_grad()
    o1 = torch.tanh(n1);o1.retain_grad()
    print(f"o1={o1}")
    # forward pass with x2
    n2 = x2@w1 + b;n2.retain_grad()
    o2 = torch.tanh(n2);o2.retain_grad()
    print(f"o2={o2}")

    loss = (y[0]-o1)**2 + (y[1]-o2)**2
    loss.backward()
    
    #print(f"dL/do1")
    print(f"{o1.grad=}") 
    dLdo1 = 2*o1 - 2*y[0]
    print(f"{dLdo1=}")

    #print(f"dL/do2")
    print(f"{o2.grad=}")
    dLdo2 = 2*o2 - 2*y[1]
    print(f"{dLdo2=}")


    #print(f"do1/db")
    do1db = 1- torch.tanh(n1)**2
    print(f"{do1db=}")

    #print(f"do2/db")
    do2db = 1- torch.tanh(n2)**2
    print(f"{do2db=}")

    #print(f"\ndL/db")
    print(f"{b.grad=}")
    # dL/db = dL/o1 * do1/db + dL/o2 * do2/db
    dLdb = dLdo1*do1db + dLdo2*do2db
    print("Gradient for bias b")
    print(f"{dLdo1*do1db.item()} + {dLdo2*do2db.item()} = {dLdb.item()=}")


    #print(f"\ndL/dw11")
    # dL/dw1 = dL/do1 * do1/dw1
    # - dL/do1 = 2*o1- 2*y[0]
    # - do1/dw1 = d(tanh(x1*w1+b))/dw1
    #       g(x) = tanh(ax+b) // g'(x) = a * tanh'(ax+b) = a * (1 - tanh(ax+b)*tanh(ax+b))
    #       a -> x1 = x1[0]
    dLdw11 = (2*o1- 2*y[0])*x1[0]*(1-torch.tanh(n1)**2)
    print(f"{dLdw11=}")

    #print(f"dL/dw12")
    dLdw12 = (2*o1- 2*y[0])*x1[1]*(1-torch.tanh(n1)**2)
    print(f"{dLdw12=}")


    #print(f"dL/dw21")
    dLdw21 = (2*o2- 2*y[1])*x2[0]*(1-torch.tanh(n2)**2)
    print(f"{dLdw21=}")

    #print(f"dL/dw22")
    dLdw22 = (2*o2- 2*y[1])*x2[1]*(1-torch.tanh(n2)**2)
    print(f"{dLdw22=}")
    print(dLdw11+dLdw21)
    print(dLdw12+dLdw22)
    print(f"{w1.grad=}")

def main():
    manualprop3()

def formulaprop(n1,o1,b,y):
    return 2*(o1-1)*(1- torch.tanh(n1)**2)*torch.tanh(b)
    


if __name__ == "__main__":
    main()


