import firedrake
import icepack
from icepack.constants import ice_density as rho_ice, water_density as rho_sw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def icepack_model(icepack_model, Q_c, V):
    """
    Creates an icepack model from the specified functionals and type.

    Args:
        icepack_model (named tuple): Dictionary containing the functional types of the ice model and the type of icepack model to use.
        Q_c (float): The rate of basal melting or freezing.
        V (float): The ice velocity.

    Returns:
        model: An icepack.model instance.
        diagnostic_solver_kwargs (dict): A dictionary of keyword arguments that can be passed to the diagnostic solver.
    """
    model_kwargs = {}
    diagnostic_solver_kwargs = {}
    if icepack_model.laws is not None:
        if "fluidity" in icepack_model.laws.keys():
            diagnostic_solver_kwargs["fluidity"] = make_fluidity_law(icepack_model.laws.fluidity.temp)
        if "log_fluidity" in icepack_model.laws.keys():
            model_kwargs["viscosity"], diagnostic_solver_kwargs["log_fluidity"] = make_log_fluidity_viscosity_law(icepack_model.laws.log_fluidity.temp, Q_c)
        if "friction" in icepack_model.laws.keys():
            model_kwargs["friction"], diagnostic_solver_kwargs["friction"] = make_friction_law(icepack_model.laws.friction, Q_c, V)

        # can implement also gravity,terminus,side_friction,penalty,continuity terms here

    if icepack_model.name == "shelf":
        print(model_kwargs)
        model = icepack.models.IceShelf(**model_kwargs)
    elif icepack_model.name == "stream":
        model = icepack.models.IceStream(**model_kwargs)
    elif icepack_model.name == "hybrid":
        model = icepack.models.HybridModel(**model_kwargs)

    return model, diagnostic_solver_kwargs


def make_log_fluidity_viscosity_law(temp,Q_c):
    """
    Creates a viscosity law as a function of the log fludiity from the given  base temperature.

    Args:
        temp (float): The temperature at which the viscosity is evaluated.
        Q_c (float): The function space.
    """
    A0 = icepack.rate_factor(firedrake.Constant(temp))
    true_lf = firedrake.Function(Q_c)

    def viscosity(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        theta = kwargs["log_fluidity"]

        A = A0 * firedrake.exp(theta)
        return icepack.models.viscosity.viscosity_depth_averaged(
            velocity=u, thickness=h, fluidity=A
        )
    
    return viscosity,true_lf

def make_fluidity_law(temp):
    """
    Creates a fluidity law from the given base temperature.

    Args:
        temp (float): The temperature at which the fluidity is evaluated.
    """
    A0 = icepack.rate_factor(firedrake.Constant(temp))
    return A0

def make_friction_law(friction,Q_c,V):
    """
    Creates a friction law from the given friction law parameters.

    Args:
        friction (named tuple): A named tuple containing the parameters for the friction law.
        Q_c (firedrake.FunctionSpace): The function space on which to interpolate the friction law.
        V (firedrake.FunctionSpace): The function space on which to interpolate the velocity.
    """
    if friction.type == "constant":
        true_friction = firedrake.Constant(friction.const)
        def const_friction(**kwargs):
            friction = kwargs["friction"]
            return friction
        return const_friction,true_friction
    else:
        raise NotImplementedError(f"Friction law {friction.type} not implemented")

def icepack_solver(solver_kwargs,model):
    """
    Creates a dictionary of keyword arguments that can be passed to the icepack solver.
    Note that due to the way icepack is implemented, the keyword arguments are not passed directly to the solver, but are instead used to create a dictionary of keyword arguments that are passed to the solver.
    Hacky workaround.

    Args:
        solver_kwargs (named tuple): A named tuple containing the parameters for the icepack solver.
        model (icepack.model): The icepack model.
    """

    side_wall_ids = solver_kwargs.get("side_wall_ids",[])
    ice_front_ids = solver_kwargs.get("ice_front_ids",[])
    dirichlet_ids = solver_kwargs.get("dirichlet_ids",[1])
    diagnostic_solver_type = solver_kwargs.get("diagnostic_solver_type","icepack")
    diagnostic_solver_parameters = {key:solver_kwargs["diagnostic_solver_parameters"][key] for key in solver_kwargs["diagnostic_solver_parameters"].keys()}
    opts = {
            "side_wall_ids":side_wall_ids,
            "dirichlet_ids":dirichlet_ids,
            "ice_front_ids":ice_front_ids,
            "diagnostic_solver_type":diagnostic_solver_type,
            "diagnostic_solver_parameters":diagnostic_solver_parameters
        }

    

    return opts


def create_initial_conditions(ics,mesh,Q_c,V,Lx,Ly):
    """
    Creates the initial conditions for the icepack model.

    Args:
        ics (named tuple): A named tuple containing the parameters for the initial conditions.
        mesh (firedrake.Mesh): The mesh on which to interpolate the initial conditions.
        Q_c (firedrake.FunctionSpace): The function space on which to interpolate the thickness.
        V (firedrake.FunctionSpace): The function space on which to interpolate the velocity.
        Lx (float): The length of the domain in the x-direction.
        Ly (float): The length of the domain in the y-direction.
    """
    logger.info("Creating initial conditions")
    h0 = profile_factory(ics.thickness.base,mesh,Lx,Ly,Q=Q_c)
    logger.info("Created initial thickness")
    ux = profile_factory(ics.vx.base,mesh,Lx,Ly,Q=V.sub(0))
    uy = profile_factory(ics.vy.base,mesh,Lx,Ly,Q=V.sub(1))

    logger.info("Created initial velocity")
    if ics.thickness.noise:
        h0 = firedrake.assemble(h0 + firedrake.interpolate(profile_factory(ics.thickness.noise,mesh,Lx,Ly,Q=Q_c),Q_c))
    if ics.vx.noise:
        ux_noise =  profile_factory(ics.vx.noise,mesh,Lx,Ly,Q=V.sub(0))
        ux = firedrake.assemble(ux + firedrake.interpolate(ux_noise,V.sub(0)))
    if ics.vy.noise:
        uy_noise =  profile_factory(ics.vy.noise,mesh,Lx,Ly,Q=V.sub(1))
        uy = firedrake.assemble(uy + firedrake.interpolate(uy_noise,V.sub(1)))
    u_guess = firedrake.interpolate(firedrake.as_vector([ux,uy]),V)
    return h0,u_guess



def profile_factory(params,mesh,Lx,Ly,Q=None):
    """
    Interpolates a flow profile based on the given parameters.

    Args:
        params (namedtuple): A named tuple containing the parameters for the flow profile.
        mesh (firedrake.Mesh): The mesh on which to interpolate the flow profile.
        Lx (float): The length of the domain in the x-direction.
        Ly (float): The length of the domain in the y-direction.
        Q (firedrake.FunctionSpace, optional): The function space on which to interpolate the flow profile. Defaults to None.

    Returns:
        firedrake.Function: The interpolated flow profile.
    """
    if mesh.geometric_dimension() == 2:
        x,y = firedrake.SpatialCoordinate(mesh)

    elif mesh.geometric_dimension() == 3:
        x,y,z = firedrake.SpatialCoordinate(mesh)

    if params.type=="linear":
        expr = params.f_in + (params.f_out - params.f_in)*x/Lx
        return firedrake.interpolate(expr,Q)
    
    elif params.type=="constant":
        expr = params.const*x/x
        return firedrake.interpolate(expr,Q)
    
    elif params.type == "converge":
        expr = -params.scale*(y-Ly/2)/Ly
        return firedrake.interpolate(expr,Q)
    
    elif params.type == "central_flowline":
        expr =  (params.f_in + (params.f_out-params.f_out)*(x)/Lx)*(1.0 + (y-0.0)*(y-Ly)/(Ly**2))
        return firedrake.interpolate(expr,Q)
    
    elif params.type == "gaussian_spike":
        expr =  params.base + params.magnitude*firedrake.exp(-((x-Lx*params.relative_location)/(Lx*params.relative_lengthscale))**2)
        return firedrake.interpolate(expr,Q)
    
    elif params.type=="GP":
        func = firedrake.Function(Q)
        h_ker = params.scale * RBF(length_scale=params.length_scale)
        gpr = GaussianProcessRegressor(kernel=h_ker)

        m = Q.ufl_domain()
        WW = firedrake.VectorFunctionSpace(m, Q.ufl_element())
        X = firedrake.interpolate(m.coordinates,WW)
        Y = X.dat.data_ro[:,:].copy()
        Y[:,1] *= (Lx/Ly)

        mvn_mean,mvn_cov = gpr.predict(Y,return_cov=True)
        eps = 1e-6
        a = np.zeros(shape = mvn_cov.shape)
        np.fill_diagonal(a,eps)
        mvn_cov += a
        mvn_mean = torch.from_numpy(mvn_mean) + params.const
        mvn_cov = torch.from_numpy(mvn_cov)
        prior = torch.distributions.MultivariateNormal(loc=mvn_mean,covariance_matrix=mvn_cov)

        sample = gpr.sample_y(Y,n_samples =1,random_state=None)
        sample = prior.sample().numpy()
        func.dat.data[:] += sample.flatten()
        return func


def create_mass_balance(mb_kwags,mesh,Q_c,Lx,Ly):
    """
    Creates the mass balance for the icepack model.

    Args:
    mb_kwags (named tuple): A named tuple containing the parameters for the mass balance.
    mesh (firedrake.Mesh): The mesh on which to interpolate the mass balance.
    Q_c (firedrake.FunctionSpace): The function space on which to interpolate the mass balance.
    Lx (float): The length of the domain in the x-direction.
    Ly (float): The length of the domain in the y-direction.
    """
    logger.info("mass balance components:")
    if len(mb_kwags.keys()) == 1:
        logger.info(mb_kwags)
        key = list(mb_kwags.keys())[0]
        mass_balance = profile_factory(mb_kwags[key],mesh,Lx,Ly,Q_c)
    elif len(mb_kwags.keys())>1:
        mass_balance = firedrake.Function(Q_c)
        for key in mb_kwags.keys():
            logger.info(mb_kwags[key])
            mass_balance = mass_balance + profile_factory(mb_kwags[key],mesh,Lx,Ly,Q_c)
        mass_balance = firedrake.interpolate(mass_balance,Q_c)
    return mass_balance
