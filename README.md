# QuantumDemos

This repository contains some notebooks for
[QuantumOps.jl](https://github.ibm.com/John-Lapeyre/QuantumOps.jl)
and [ElectronicStructure.jl](https://github.ibm.com/John-Lapeyre/ElectronicStructure.jl).

* The [notebooks](./notebooks) are a mixture of demonstrations, documentation, etc.

* Most of the notebooks are generated from [text files](./source/) via [Literate.jl](https://github.com/fredrikekre/Literate.jl).


## Setting up the environment to generate or run these notebooks

These notebooks are developed using a Python virtual environment, and Julia projects.
Here is a way to set it up.

```shell
shell> python -m venv ./venv # to create the virtual environment
shell> source ./venv/bin/activate
shell> pip install -r requirements.txt
shell> export PYCALL_JL_RUNTIME_PYTHON=`which python` # tell PyCall which python to use
```
The last two steps can be done instead via `source ./bin/activate_all.sh`.
Then start julia
```julia
julia>
shell> cd ./source # switched to shell mode
(@v1.7) pkg> activate .  # activate project
julia> include("literate_funcs.jl") # load the function to make notebooks
julia> notebook("./quantum_ops_intro.jl") # to process Literate document into notebook
```
