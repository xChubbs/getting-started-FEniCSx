# Learning FEniCSx
This is small project tries to create a small interactive tutorial in which we can explore the framework and get started with FEniCSx.
The idea behind this series of files is to, step by step, guide the reader through the basics and how to incorporate already existing code and files into FEniCSx.

The project is organized as follows:
~~~
Projects/
├── 1_learning_fenics.py
├── 2_loading_gmsh_msh.py
├── 3_hdg_meets_fenics.py
├── README.md
├── documentation/
│   ├── FEniCS-book.pdf
│   └── Solving PDE-python.pdf
└── include/
    └── gmsh/
        ├── cubeD.geo
        ├── cubeD0.msh
        ├── cubeD1.msh
        ├── cubeD2.msh
        ├── cubeD3.msh
        ├── cubeD4.msh
        ├── cubeD5.msh
        └── import_msh.py
~~~

## Installation
It's recommended to install FEniCSx using the official installation guide provided by the FEniCSx team, I personally recommend using the `conda` environment approach using `python 3.11.x` since you can easily manage dependencies, and avoid version conflics through the files. You can find the installation guide at [https://fenicsproject.org/download/](https://fenicsproject.org/download/). Although, some additional packages are needed for the graphics that are missing on the official guide, remember to install this packages using the `pip` linked into your `conda` environment:
~~~
pip install trame-vtk trame-vuetify ipywidgets
~~~

## Tutorial Overview

### 1. Basic Poisson Equation (`1_learning_fenics.py`)
This script demonstrates how to solve a standard Poisson equation ($-\Delta u = f$) on a 2D rectangular domain with an standard FEM implementation:
- Lagrange elements (degree 2).
- Dirichlet boundary conditions.
- Visualization using `pyvista`.

### 2. Loading External Meshes (`2_loading_gmsh_msh.py`)
Learn how to bridge GMSH with FEniCSx. This tutorial covers:
- Reading `.msh` files using `dolfinx.io.gmsh`.
- Defining integration measures (`dx`, `ds`) based on mesh markers.
- Validating the loaded mesh by computing volume and area.

### 3. HDG Method Implementation (`3_hdg_meets_fenics.py`)
An advanced deep-dive into the Hybridizable Discontinuous Galerkin (HDG) method. It features:
- Mixed function spaces (Volume & Skeleton).
- Manual assembly of blocked systems.
- Custom norms for error calculation.
- PETSc KSP solver configuration for large systems.

## Documentation
For a deeper theoretical understanding, refer to the PDF files in the `documentation/` directory, which include the official FEniCS book and a guide on solving PDEs with Python.
