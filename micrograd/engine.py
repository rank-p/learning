from graphviz import Digraph
import math
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def exp(self):
        x = math.exp(self.data)
        out = Value(x, (self,), 'exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f"**{other}")

        def _backward():
            self.grad += out.grad * (other * (self.data ** (other-1)))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1-t**2)
        out._backward = _backward
        return out

    def _topo(self):
        visited = set()
        topo = []
        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for c in n._prev:
                    build_topo(c)
                topo.append(n)
            return topo
        return build_topo(self)

    def backward(self):
        self.grad = 1
        topo = self._topo()
        for n in reversed(topo):
            n._backward()

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

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

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label= "{ %s |  data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape="record")
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


if __name__ == "__main__":
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813715870195432, label="b")
    x1w1 = x1*w1; x1w1.label = "x1w1"
    x2w2 = x2*w2; x2w2.label = "x2w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label="x1w1x2w2"
    n = x1w1x2w2 + b; n.label = "n"
    o = n.tanh(); o.label ="o"
    o.backward()
    dot = draw_dot(o)
    dot.render('graph', view=True)

