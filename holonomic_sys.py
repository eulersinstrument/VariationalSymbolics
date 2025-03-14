import sympy as sym
import numpy as np
from sympy import sin, cos, Matrix, Symbol, symbols
from sympy.physics.mechanics import dynamicsymbols
from typing import Callable


class Holonomic_System:
    #TODO: Figure out how to include frictional forces arrising from constraints. For example, if a bead moves along 
    #a wire at a certain velocity and position, what is the resulting frictional force. Obviously, this problem is solved
    #once one finds the normal force in terms of the state of the system.

    def __init__(self,
                f_cart: Matrix, 
                m: Matrix, 
                q: Matrix, 
                cart_coor: Matrix , 
                u: Matrix, 
                phi: Callable[[Matrix],Matrix]
        ):
        """Automates the process of deriving the equations of motion for a holonomic system. By calling control_init,
        one can automatically generate the canonical A and B matrices (dynamics and control matrices, respectively) used 
        in various control engineering paradigms.

        Args:
            f_cart (Matrix): Sum of forces on each particle. shape = (n,3) 
            m (Matrix): Mass of each particle. shape = (n,1)
            q (Matrix): symbols corresponding to the generalized coordinates. shape = (dof,1)
            cart_coor (Matrix): symbols corresponding to the cartesian coordinates:
            (x_1, x_2, ..., x_n, y_1, ..., y_n, z_1, ...). shape = (3*n,1)
            u (Matrix): symbols corresponding to the control variables. shape = (k, 1)
            phi (Callable[[Matrix],Matrix]): function which transforms the generalized coordinates into the 
            corresponding cartesian coordinates of the system. shapes = (dof,1) -> (3*n,1)
        """
        
        self.t = symbols("t") #time 
        self.dof = q.shape[0] #degrees of freedom
        self.n = f_cart.shape[0] #number of point masses ( f_cart.shape = (num_point_masses, 3) )
        self.u = u 
        self.q = q
        self.q_dot = _D(self.q, self.t) #first time derivative of q d_dt(q)
        self.q_ddot = _D(self.q_dot, self.t)#second time derivative of q  d^2_dt^2(q)


        self.f_gen = self._compute_f_gen(_transform_f_cart(f_cart, cart_coor, phi(q)), phi(q)) #generalized forces
        self.T = _transform_T(m, phi, self.t)(q)
        
        self.eqs_motion = sym.trigsimp(self._gen_eqs_motion())

    
    def _compute_f_gen(self, f_cart: Matrix, phi_sym: Matrix):
        """Computes the generalized forces corresponding to each generalized coordinate, using the following formula
        derived from D'Alembert's Principle of virtual work.
        F_qi (generalized force corresponding to qi) = F_j.dot(∂r_j/∂qi) (Einstein convention)
        Args:
            f_cart (Matrix): Sum of forces on each particle. shape = (n,3) 
            phi_sym (Matrix): positions of the point masses, described in terms of the generalized coordinates.
            shape = (3*n,1)
            

        Returns:
            (Matrix): shape = (1,n), since it must coincide with the shape of ∂T/∂q.shape() = (1,n)
        """
        

        #rearranges phi_sym, so that it is a matrix whose ith column is the position of the ith point mass
        split_points = [0,self.n, 2*self.n, 3*self.n]   
        r = Matrix([phi_sym[split_points[i]: split_points[i+1]] for i in range(3)]).T


        return  Matrix(
            [np.sum(np.multiply(f_cart, _D(r, self.q[i]))) for i in range(self.dof)]
        ).T
        
    
    def _gen_eqs_motion(self):
        """D'Alemberts principle is used here, instead of the Lagrangian action minimization principle. ie
        d_dt(∂T/∂d_dt(q)) - ∂T/∂q - f_gen = 0"""
 
        return Matrix([
            _D(_D(self.T,self.q_dot), self.t) - _D(self.T, self.q) - self.f_gen
        ])
            
    def _linearize_eqs_motion_about(self, state0: dict[Symbol,float]) -> list[Matrix,Matrix]:
        """Converts self.eqs_motion, into the dynamics and control matrices A and B, respectively. The process is as follows:
        1) Find M(state),f , where M(state)(q_ddot) = f is equivalent to the equations of motion, which was previously computed.
        2) find ∂M(state).inv()*f/∂state (state0) = A as well as ∂M(state).inv()*f/u = B
        3) add the dependence between position and velocity to A and B.

        Args:
            state0 (dict[Symbol,float]): state about which the linearization should take place

        Returns:
            list[Matrix,Matrix]: (A,B), where A.shape = (2*dof, 2*dof) and B.shape = (2*dof, 1)
        """
        
        state_vars = Matrix([*self.q, *self.q_dot])
        
        M,f = sym.linear_eq_to_matrix(self.eqs_motion, list(self.q_ddot)) 

        #evaluates M about the state0
        M_0 = M.subs(state0)   
        M_0_inv = M_0.inv()
        


        D_xi_state_rep_M = [_D(M,xi).subs(state0) for xi in state_vars]
        A_incomplete = (Matrix([((-M_0_inv*D_xi_state_rep_M[i]*M_0_inv)*f).T for i in range(self.dof*2)]).T 
                        + (M_0_inv*(_D(f,Matrix([*self.q,*self.q_dot])))).subs(state0)) 
        B_incomplete = M_0_inv*_D(f,self.u) 


        #returns dynamics and control matrix (respectively)
        return sym.simplify(_complete_dynamics_matrix(A_incomplete)), sym.simplify(_complete_control_matrix(B_incomplete))



    def control_init(self, state0: dict[Symbol,float]) -> list[Matrix,Matrix]:
        """Returns the dynamics and control matrices about state0.


        Args:
            state0 (dict[Symbol,float]): state about which the linearization should take place

        Returns:
            list[Matrix,Matrix]: (A,B), where A.shape = (2*dof, 2*dof) and B.shape = (2*dof, 1).
            Additionally, A and B are structured so that all qi_dot come before q_iddot:
            (q1_dot, q_2dot, ..., q_dofdot, q_1ddot, ... q_dofddot)
        """

        return self._linearize_eqs_motion_about(state0)
    

def _transform_T(m:Matrix ,phi: Callable, t: Symbol):
    """expresses the kinetic energy of a system of point masses in terms of the 
    generalized coordinates (m could be a vector)"""

    mx3 = Matrix([m for i in range(3)]) 

    dphi_dt = lambda q: _D(phi(q),t)
    return lambda q: 1/2 * (dphi_dt(q).multiply_elementwise(mx3).T*dphi_dt(q))

def _transform_f_cart(f_cart, cart_coor, phi_sym):
    """expresses all the """
    return f_cart.subs(dict(zip(cart_coor, phi_sym)))

def _D(exp,x):
    """Takes the derivative of the expression with respect to the variable x. exp and x must either be sym vectors or
    scalar expressions."""
    
    if type(x) != Matrix: #the jacobian method for matrix requires the variables to be held in a list
        return sym.diff(exp,x)
    else: return exp.jacobian(x)

def _complete_dynamics_matrix(A):
        """Adds the relationship dx2i_dt = x2i+1 to the dynamics matrix A"""
        dof = A.shape[0]

        return Matrix([
            [ #the rate of change of the generalized coordinate directly precedes the rate of change of its velocity
                Matrix([1 if j == i+dof else 0 for j in range(dof*2)]) if i < dof 
                else A[i-dof,:].T
                for i in range(dof*2)
            ]
        ]).T

def _complete_control_matrix(B):
        """adds zero rows corresponing to the position state variables"""
        dof = B.shape[0]
        num_controls = B.shape[1]
        return Matrix([
            [ 
                sym.zeros(num_controls) if i < dof #position rows
                else B[i-dof,:] #velocity rows
                for i in range(dof*2)
            ]
        ]).T



    






