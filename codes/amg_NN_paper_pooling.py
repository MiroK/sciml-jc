import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from dolfin import *
import numpy as np

mesh = UnitSquareMesh(32, 32)

kappa_values = [1, 1000, 10, 100]

S = FunctionSpace(mesh, 'DG', 0)
kappa = Function(S)
kappa.vector().set_local(np.random.choice(kappa_values,
                                          kappa.vector().local_size()))

V = FunctionSpace(mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)

a = inner(kappa*grad(u), grad(v))*dx
A = assemble(a)

csr = as_backend_type(A).mat().getValuesCSR()
coo = csr_matrix(csr[::-1]).tocoo()

row, col, val = coo.row, coo.col, coo.data

n = V.dim()
# The compression algorithm
m = 5
assert 0 < m <= n

q = n//m
p = n % m
t = p*(q+1)

ii = np.where(row < t, row//(q+1), (row-t)//q + p)
jj = np.where(col < t, col//(q+1), (col-t)//q + p)

V = np.zeros((m, m))
C = np.zeros_like(V)

for i, j, val in zip(ii, jj, val):
    V[i, j] += val
    C[i, j] += 1

# In logspace
kappa.vector().set_local(np.log10(kappa.vector().get_local()))
plot(kappa)

plt.figure()
plt.pcolor(C)
plt.colorbar()

plt.figure()
plt.pcolor(V)
plt.colorbar()

plt.figure()
plt.spy(coo)

plt.show()
