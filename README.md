# QuantumDemos

This repository contains some notebooks for
[QuantumOps.jl](https://github.ibm.com/John-Lapeyre/QuantumOps.jl)
and [ElectronicStructure.jl](https://github.ibm.com/John-Lapeyre/ElectronicStructure.jl).

* The notebooks are a mixture of demonstrations, documentation, etc.

* Most of the notebooks are generated from [text files](./source/) via [Literate.jl](https://github.com/fredrikekre/Literate.jl).


## Setting up the environment to generate or run these notebooks

These notebooks are developed using a Python virtual environment, and Julia projects.
Here is a way to set it up.

* `python -m venv ./venv` to create the virtual environment
* `source ./venv/bin/activate`, or the appropriate command for your shell
*  Do
```shell
export PYCALL_JL_RUNTIME_PYTHON=`which python`
```
to tell `PyCall` to use the virtual environment.
* The last two steps can be done via `source ./bin/activate_all.sh`
* `cd ./source`
* Start julia and do `import Pkg; Pkg.activate(".")` to activate the project



