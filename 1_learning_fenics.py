##################################################################
# author : @alujan
# file : learning_fenics.py
##################################################################
# FEniCS self tutorial before working with external
# colleagues, this is a reproduction of usable book's
# codes referenced on the documentation.
##################################################################
# @about
# The idea of the file it's to dissect the example
# provided to solve the laplace equation for latter
# implementation of the SAYAS example of use.
# For further informatio  of the material used for
# reference visit:
# https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html
##################################################################

# %% 0. Importing modules
import ufl
import pyvista
import numpy as np

from mpi4py             import MPI
from dolfinx             import fem, mesh, plot, default_scalar_type
from petsc4py.PETSc      import ScalarType
from dolfinx.fem.petsc   import LinearProblem

# %% 1. Context definition

# - Definition of example domain
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,                # Communicator for parallel behaviour
    points=((0.0, 0.0), (1.0, 1.0)),    # Definition of two domain points
    n=(8, 8),                           # Number of elements on each direction
    cell_type=mesh.CellType.triangle,   # Cell type for each element
)

# - Function space definition
k = 1                       # Degree of Lagrange polynomials
V = fem.functionspace(
    msh, ("Lagrange", k))   # Here's used Lagrange polynomials & degree k >= 0

# - Definition of dimentions of each type of variable
d2 = msh.topology.dim    # Dimention of volume variables
d1 = d2 - 1              # Dimention of face variables

# - Connectivity between faces and volume elements
msh.topology.create_connectivity(d1, d2)

# Dirichlet boundary conditions faces
#   We need to "manually" define the Boundary faces by locating the values
#   closest to the coordinates in which we defined the rectangle

# - Function selector for the closest points to the "outside" faces.
def close_to_frontier(x : np.ndarray) -> int:
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | \
           np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

# - Location of the Boundary faces
#   It's important to remark that the same process could have been done
#   using the function mesh.exterior_facet_indices(msh.topology)
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=close_to_frontier,
)

# - Locate of DOF for Dirichlet conditions
#   The same here it's achieved by using:
#   mesh.exterior_facet_indices(msh.topology)
dofs = fem.locate_dofs_topological(V=V,     # Function space used
                         entity_dim=d1,     # Dimention of the faces identities
                        entities=facets)    # indexes of the faces found


# %%- Imposition of Dirichlet BC
uD = fem.Function(V)                        # Function for Dirichlet BC
uD.interpolate = lambda x: ScalarType(0)    # Definition of function values

bc = fem.dirichletbc(value=uD, dofs=dofs)   # Imposition of values on DOF's

# %% 2. Definition of variational problem

# - Definition of functions involved
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# - Retrieve of spacial coordinates for points
x = ufl.SpatialCoordinate(msh)

# - Source (f) definition
#   >> If looking for a constant definition use:
#       fem.Constant(domain, default_scalar_type(0))
#       (change the 0 to the constant value)
#   >> If going for a function definition consider the function:
#       10 \exp(-\frac{(x - 0.5)^2}{0.02})
#       In this case we define f as:
#       f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
f = np.pi**2 * (ufl.sin(np.pi*x[0]) + ufl.sin(np.pi*x[1]))

# - Defintion of bilinear form (a) and linear functional (F)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# %% 3. Solution of variational problem
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="demo_poisson_",   # PETSc backend definition
    petsc_options={                         # PETSc backend options
        "ksp_type": "preonly",              # - Load precondition only once
        "pc_type": "lu",                    # - Uses LU factorization
        "ksp_error_if_not_converged": True, # - Error if not converged
    },
)
uh = problem.solve()                        # Solution "(A \ b)"

# %% 4. Solution validation
# - Definition of exact solution
V2  = fem.functionspace(msh, ("Lagrange", k+1))   # Definition of solution space
uex = fem.Function(V2, name="u_exact")            # Definition of exact solution

# - Exact solution interpolation function
uex.interpolate(lambda x: np.sin(np.pi * x[0]) + np.sin(np.pi * x[1]))

# - Definition of L2 error
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)

error_local = fem.assemble_scalar(L2_error)         # Assembly of all local L2 errors

error_L2 = np.sqrt(
    msh.comm.allreduce(error_local, op=MPI.SUM))    # Computation of global L2 error

error_max = np.max(np.abs(uD.x.array - uh.x.array)) # Max global error

if msh.comm.rank == 0:                              # Only print the error on one process
    print(f"Error_L2 :  {error_L2:.2e}" )
    print(f"Error_max : {error_max:.2e}")

# %% 5. Graphing of solution
cells, types, x = plot.vtk_mesh(V)                  # Plot of mesh
grid = pyvista.UnstructuredGrid(cells, types, x)    # Creation of pyvista grid
grid.point_data["u"] = uh.x.array.real              # Assigment to real
grid.set_active_scalars("u")                        # Active mesh scalar
plotter = pyvista.Plotter()                         # Pyvista plotter
warped = grid.warp_by_scalar()                      # wrap scalar values

plotter.add_mesh(warped,
                show_edges=True,
                show_scalar_bar=True,
                smooth_shading=True)
plotter.show_axes_all()
plotter.show_grid()
plotter.show()
