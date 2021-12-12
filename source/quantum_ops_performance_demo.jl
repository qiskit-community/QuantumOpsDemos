# # `QuantumOps` performance intro

# This notebook introduces some of `QuantumOps` with an emphasis on performance aspects.

# Tell `PyCall` to use our virtual Python environment
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")

## import all default symbols for interactive demo
using QuantumOps
using QuantumOps: AbstractOp
using BenchmarkTools
import LinearAlgebra
import SparseArrays

## We also import I, X, Y, Z for convenience
using QuantumOps.Paulis: I, X, Y, Z
#----------------------------------------------------------------------------

# ### Pauli
#
# These, `I, X, Y, Z`, are bound to instances of the `Pauli` type, representing single-qubit operators.

(I, X, Y, Z) == Pauli.((0, 1, 2, 3)) ## The '.' broadcasts over the following elements.
#----------------------------------------------------------------------------

# Julia has a large number of standard interfaces and functions for numeric and algebraic types. I follow these when possible. For example, `Matrix` is used to construct a dense, heap-allocated matrix from an object. So I defined a method for it.

print(Matrix.(Pauli.(0:3)))
#----------------------------------------------------------------------------

# `Pauli` is in this type hierarchy.

Pauli <: AbstractPauli <: AbstractOp
#----------------------------------------------------------------------------

# Only a very small amount of code depends on the internals of `Pauli`.
# Almost everything is coded against `AbstractOp` and `AbstractPauli`. The developer almost never encounters the implementation of `Pauli`, and the user never does.
# But, if you want, you can see it this way.

dump(X)
#----------------------------------------------------------------------------


# The notation `X[i, j]` calls `getindex(X, i, j)` which, for `AbstractPauli`, looks up the elements in stack allocated arrays.
# This is faster than indexing into a heap allocated (that is, every-day, dynamic) array:

m = rand(2, 2) # Ordinary `Matrix`
@btime $m[1, 1];  # @btime is like %timeit. The "$" tests how it would perform in a compiled function
#----------------------------------------------------------------------------

@btime X[1, 1];  ## This includes time to look up what matrix corresponds to `X`
#----------------------------------------------------------------------------

# Some linear algebra has been implemented.

# In Julia, multiplying two small matrices is faster than multiplying two numpy matrices is Python. The Python call includes an overhead.
# Multiplying `QuantumOps.Paulis.Pauli` by a `Matrix` is even a bit faster because looking up elements of a `Pauli` is faster.

mY = Matrix(Y) # convert Y to a `Matrix`

@btime Y * $m;
#----------------------------------------------------------------------------
@btime $mY * $m;

# Another example:

@btime LinearAlgebra.eigvals(Z)
#----------------------------------------------------------------------------

# `20`ns is the time required to copy the array of eigenvalues.

# ### `PauliTerm`
#
# `PauliTerm` represents a tensor product of Pauli operators (or a single one) and keeps track of a coefficient, including a phase.

# ### Compare `PauliTerm` with qiskit

using PyCall
qi = pyimport("qiskit.quantum_info");
#----------------------------------------------------------------------------

# Here, we compare multiplication of two Pauli strings with both libraries. We see how the time scales with string length.

function get_julia_python_terms(n_qubits)
    xj = PauliTerm(rand(Pauli, n_qubits))
    yj = PauliTerm(rand(Pauli, n_qubits))
    xp = qi.random_pauli(n_qubits)
    yp = qi.random_pauli(n_qubits)
    return (xj, yj, xp, yp)
end

n_qubits = 10
(xj, yj, xp, yp) = get_julia_python_terms(n_qubits)

## `QuantumOps`
@btime $xj * $yj
#----------------------------------------------------------------------------

## qiskit
@btime $xp.compose($yp)   ## @btime gives same times as %timeit in python cli
#----------------------------------------------------------------------------

# #### For mulitplying 10-qubit Pauli strings, `QuantumOps` is about 300 times faster than qiskit.

julia_time = @belapsed $xj * $xj
qiskit_time = @belapsed $xp.compose($yp)

qiskit_time / julia_time
#----------------------------------------------------------------------------

# Asymptotically, qiskit is about three times faster than `QuantumOps`. But, it takes a while to get there. For 1000-qubit strings `QuantumOps` is still eight times faster. I have some ideas regarding why python is faster than Julia here, but I am not at all sure. Also, there is a big, ~12 micro-s constant term in the python times. It might be worth trying to reduce this.

n_qubits = 1000
(xj, yj, xp, yp) = get_julia_python_terms(n_qubits)

julia_time = @belapsed $xj * $xj
qiskit_time = @belapsed $xp.compose($yp)

qiskit_time / julia_time
#----------------------------------------------------------------------------

# Here are $10^4$ qubits. Julia is still faster, but they are comparable.

n_qubits = 10^4
(xj, yj, xp, yp) = get_julia_python_terms(n_qubits)

julia_time = @belapsed $xj * $xj
qiskit_time = @belapsed $xp.compose($yp)

qiskit_time / julia_time
#----------------------------------------------------------------------------

# ### `PauliSum`

# A `PauliSum` represents a sum of `PauliTerm`s, sorted in a canonical order.

n_qubits = 10
n_terms = 10
ps = PauliSum(rand(Pauli, (n_terms, n_qubits)), randn(n_terms))
#----------------------------------------------------------------------------

x = ps[5]
(x, typeof(x))
#----------------------------------------------------------------------------

# `add!` adds a `PauliTerm` in place. It does a sorted search to find the correct location.

add!(ps, -x)  ## add the additive inverse of a term
#----------------------------------------------------------------------------

# The length of the sum is now 9 rather than 10.

length(ps)
#----------------------------------------------------------------------------

n_qubits = 10
n_terms = 10
ps = PauliSum(rand(Pauli, (n_terms, n_qubits)), randn(n_terms));
size(ps)
#----------------------------------------------------------------------------

x = copy(ps[1])
#----------------------------------------------------------------------------

@btime add!($ps, $x);
#----------------------------------------------------------------------------

# That seems a bit slow.

# ### Multiplying Pauli Sums. Compare with Qiskit

# `qiskit.quantum_info` and `QuantumOps` are comparable in performance when multiplying Pauli sums;
#  at least over some range of parameters.

function get_julia_python_sums(n_qubits, n_terms)
    xj = rand_op_sum(Pauli, n_qubits, n_terms)
    yj = rand_op_sum(Pauli, n_qubits, n_terms)
    xp = qi.SparsePauliOp(qi.random_pauli_list(n_qubits, n_terms))
    yp = qi.SparsePauliOp(qi.random_pauli_list(n_qubits, n_terms))
    return (xj, yj, xp, yp)
end

#-

n_qubits = 10; n_terms = 100

(xj, yj, xp, yp) = get_julia_python_sums(n_qubits, n_terms)
julia_time = @belapsed $xj * $xj
qiskit_time = @belapsed $xp.compose($yp).simplify()

qiskit_time / julia_time

#----------------------------------------------------------------------------

n_qubits = 10; n_terms = 2000

(xj, yj, xp, yp) = get_julia_python_sums(n_qubits, n_terms)
julia_time = @elapsed xj * xj
qiskit_time = @elapsed xp.compose(yp).simplify()

qiskit_time / julia_time

#----------------------------------------------------------------------------

n_qubits = 100; n_terms = 2000

(xj, yj, xp, yp) = get_julia_python_sums(n_qubits, n_terms)
julia_time = @elapsed xj * xj
qiskit_time = @elapsed xp.compose(yp).simplify()

qiskit_time / julia_time

# #### Pauli decomposition
#
# Construct the Pauli decomposition of a matrix.

m = rand(4, 4)
s = PauliSum(m)
#----------------------------------------------------------------------------

# Check that the decomposition is correct.

m ≈ Matrix(s)
#----------------------------------------------------------------------------

# Doing this decomposition is exponentially expensive. Here we compare the performance of Qiskit QI vs. QuantumOps.

n = 7
m = rand(2^n, 2^n);
julia_time = @belapsed PauliSum($m)
#----------------------------------------------------------------------------

qi_op = qi.Operator(m)
qi_time = @elapsed qi.SparsePauliOp.from_operator(qi_op)
#----------------------------------------------------------------------------

# Ratio of times to do Pauli decomposition for random 7-qubit matrix

qi_time / julia_time
#----------------------------------------------------------------------------

# ### Parametric types and composability

# #### Z4Group
#
# I implemented a type `Z4Group` that represents `(i, -1, -i, 1)`. This can be used to represent the Pauli group. The type `Z4Group` becomes part of the type of the term, which aids the compiler in devirtualizing and inlining.

t = PauliTerm(:XXY, Z4Group(im))
(t, typeof(t))
#----------------------------------------------------------------------------

v = PauliTerm(:ZXZ, Z4Group(1))
#----------------------------------------------------------------------------

t * v
#----------------------------------------------------------------------------

# #### Z4Group0
#
# More interesting is `Z4Group0` which is `Z4Group` augmented by another `Bool` representing zero. This can represent `(0, im, -im, 1, -1)`. It supports multiplication of elements, but is only closed under addition where at least one operand is `0`. It will error if you don't respect this. This quasi-algebra is enough to represent and compute kronecker products of Pauli matrices. The structure is this

dump(Z4Group0(1))
#----------------------------------------------------------------------------

# Note that this is a nested composite type. Nontheless an array of these is packed, with each element taking three bytes. Here is a packed two-dimensional array of `Z4Group0`.

a = rand(Z4Group0, (3,5))
a
#----------------------------------------------------------------------------

sizeof(a)  ## (3 x 5) x 3 bytes
#----------------------------------------------------------------------------

# We see that computation with `Z4Group0` can be as fast as or faster than `Complex{Int}`.

a = rand(Z4Group0, 10^5);
@btime reduce(*, a)
#----------------------------------------------------------------------------

anum = [convert(Complex{Int}, x) for x in a];
@btime reduce(*, anum)
#----------------------------------------------------------------------------

# I use `Z4Group0` in Kronecker products.

kron([Z4Group0.(m) for m in Matrix.([X, Y, Z])]...)
#----------------------------------------------------------------------------

operators = rand(Pauli, 4)
print(operators)
#----------------------------------------------------------------------------

mats = Matrix.(operators)
z40mats = [Z4Group0.(m) for m in mats];

size(kron(mats...))
#----------------------------------------------------------------------------

@btime kron($mats...);
#----------------------------------------------------------------------------

@btime kron($z40mats...);
#----------------------------------------------------------------------------

# Here, the time to do the calcuations with usual 16-byte complex numbers is the same. But, when converting a `PauliSum` to a matrix I use `ThreadsX.sum` over the terms, which is a dropin replacement for `sum` that does intelligent threading. When I use `Z4Group0` I get a significant improvement in performance, perhaps because of fewer cache misses.

# ### sympy

@pyimport sympy
(x, t) = sympy.symbols("x t")
#----------------------------------------------------------------------------

# We use a symbolic coefficient

term = PauliTerm("XXYZ", x + t)
#----------------------------------------------------------------------------

term^3
#----------------------------------------------------------------------------

# The type of coefficient is encoded in the type of the `PauliTerm`.

typeof(term)
#----------------------------------------------------------------------------

# #### `Symbolics`
#
# This is another symbolic libarary.

# The following is disabled because of errors due to changes in packages.

# using Symbolics
# #----------------------------------------------------------------------------

# @variables a b c;
# #----------------------------------------------------------------------------

# # Create a `PauliSum` with symbolic coefficients

# term1 = PauliTerm("XZ", a + b)
# term2 = PauliTerm("ZX", b + c)

# psum = term1^3 + term2
# #----------------------------------------------------------------------------

# # We convert the `PauliSum` with symbolic coefficients to a `Matrix`.
# # No additional code is necessary to support this feature.

# symmat = Matrix(psum)
# #----------------------------------------------------------------------------
