import firedrake
import icepack
from icepack.constants import ice_density as rho_ice, water_density as rho_sw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch
import numpy as np
import logging



def reg_smooth(func0,alpha,sd):
    """
    Use the built-in icepack smoothing for a function with L2 loss and smoothness regularization.

    Args:
    func0 (firedrake.Function): The function to be smoothed.
    alpha (float): The regularization constant.
    sd (float or firedrake.Function): The standard deviation of the Gaussian noise.
    """
    func = func0.copy(deepcopy=True) 
    alpha = firedrake.Constant(alpha)
    J = 0.5 * (((func - func0)/sd) ** 2 + alpha ** 2 * firedrake.inner(firedrake.grad(func), firedrake.grad(func))) * firedrake.dx
    F = firedrake.derivative(J, func)
    firedrake.solve(F == 0, func)
    return func

def create_fwd_model(inverse_param,solver,u_obs,h_obs,diagnostic_solver_kwargs):
    """
    Create a simulation object for a given icepack StatisticsProblem.

    Args:
    inverse_param (str): The name of the parameter to be estimated.
    solver (icepack.solvers.FlowSolver): The icepack solver.
    u_obs (firedrake.Function): The observed velocity.
    h_obs (firedrake.Function): The observed thickness.
    diagnostic_solver_kwargs (dict): A dictionary of keyword arguments that can be passed to the diagnostic solver.
    """
    kwargs = diagnostic_solver_kwargs.copy()
    kwargs.pop(inverse_param)
    def fwd_sim(x):
        kwargs[inverse_param] = x
        u_estimate = solver.diagnostic_solve(
            velocity=u_obs,
            thickness=h_obs,
            **kwargs,
        )
        return u_estimate
    return fwd_sim

def create_loss(loss,u_obs,h_obs,area):
    if loss.type == "L2":
        def loss_functional(u):
            delu = u - u_obs
            return (0.5 / area) * ((delu[0]/1.0)**2+(delu[1]/1.0)**2) * firedrake.dx
    else:
        raise NotImplementedError(f"Loss type {loss.type} not implemented")
    return loss_functional    


def create_reg(reg,u_obs,h_obs,area):
    if reg.type == "smooth":
        reg_const = reg.const
        def reg_functional(theta):
            L = firedrake.Constant(reg_const)
            return (0.5 / area) * (L)**2 * (firedrake.inner(firedrake.grad(theta), firedrake.grad(theta))) * firedrake.dx
    else:
        raise NotImplementedError(f"Regularization type {reg.type} not implemented")
    return reg_functional