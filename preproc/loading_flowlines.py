import hydra
import firedrake
import icepack
from icepack.constants import ice_density as rho_ice, water_density as rho_sw
from preproc.flow_models import icepack_model,icepack_solver
from preproc.inverse import reg_smooth,create_fwd_model,create_loss,create_reg
from preproc.utils.proj import trunc,get_flowline,output_regular,project_onto_dir
from preproc.utils.plot import tripcolor,ip_streamplot,plot_mesh,plot_transect,plot_all_1d
from preproc.utils.misc import get_project_root
from omegaconf import OmegaConf
import logging
from pathlib import Path
import numpy as np
from scipy.interpolate import splev, splprep
import rasterio
from icepack.statistics import (
    StatisticsProblem,
    MaximumProbabilityEstimator,
)

log = logging.getLogger(__name__)
workspace = get_project_root()
def load_mesh(cfg):
    try:
        mesh = firedrake.Mesh(str(Path(workspace,"data",cfg.shelf,cfg.mesh.mesh_file).absolute()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Mesh file {cfg.mesh_file} not found in {workspace}/data/{cfg.shelf}")
    
    mesh_family = cfg.mesh.get("mesh_family","CG")
    mesh_degree = cfg.mesh.get("mesh_degree",2)
    if cfg.mesh.dim == 2:
        Q = firedrake.FunctionSpace(mesh,mesh_family,mesh_degree)
        V = firedrake.VectorFunctionSpace(mesh,mesh_family,mesh_degree)
        return mesh,Q,V
    elif cfg.mesh.dim == 3:
        raise NotImplementedError("3D mesh loading not yet implemented")

def load_flowline(cfg):
    try:
        mesh1d = firedrake.Mesh(str(Path(workspace,"data",cfg.shelf,cfg.flowline.embedded_mesh.mesh_file).absolute()),dim=2)
    except FileNotFoundError:
        raise FileNotFoundError(f"Mesh file {cfg.flowline.embedded_mesh.mesh_file} not found in {workspace}/data/{cfg.shelf}")
    embedded_mesh_family = cfg.flowline.embedded_mesh.get("mesh_family","CG")
    embedded_mesh_degree = cfg.flowline.embedded_mesh.get("mesh_degree",1)
    V_funcspace = firedrake.VectorFunctionSpace(mesh1d,family=embedded_mesh_family,degree=embedded_mesh_degree,dim=2)
    Q_funcspace = firedrake.FunctionSpace(mesh1d,family=embedded_mesh_family,degree=embedded_mesh_degree)
    x = mesh1d.coordinates.dat.data_ro[:,0]
    y = mesh1d.coordinates.dat.data_ro[:,1]
    coordinates,dist = get_flowline(x,y,**cfg.flowline.smoothing)
    tck,u = splprep(coordinates.T, u=dist,k=3)
    tangent = np.array(splev(u,tck,der=1)).T
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    normal = np.vstack([-tangent[:,1],tangent[:,0]]).T
    normal /= np.linalg.norm(normal,axis=1)[:,None]
    return mesh1d,Q_funcspace,V_funcspace,coordinates,dist,tangent,normal

def load_fields(cfg,Q_c,V):
    try:
        thickness = rasterio.open(f'netcdf:{str(Path(workspace,"data",cfg.data.thickness.fname).absolute())}:thickness', 'r')
    except FileNotFoundError:
        raise FileNotFoundError(f"Thickness file {cfg.data.thickness.fname} not found in {workspace}/data/{cfg.shelf}")
    h_obs = icepack.interpolate(thickness,Q_c)
    try:
        fname = str(Path(workspace,"data",cfg.data.velocity.fname).absolute())
        vx = rasterio.open(f'netcdf:{fname}:VX', 'r')
        vy = rasterio.open(f'netcdf:{fname}:VY', 'r')
        stdx = rasterio.open(f'netcdf:{fname}:ERRX', 'r')
        stdy = rasterio.open(f'netcdf:{fname}:ERRY', 'r')
    except FileNotFoundError:
        raise FileNotFoundError(f"Velocity file {cfg.data.velocity.fname} not found in {workspace}/data/{cfg.shelf}")
    u_obs = icepack.interpolate((vx,vy),V)
    sigma_x = icepack.interpolate(stdx,Q_c)
    sigma_y = icepack.interpolate(stdy,Q_c)

    if cfg.data.thickness.smooth:
        log.info("Smoothing thickness")
        h = reg_smooth(h_obs,cfg.data.thickness.smoothing_alpha,cfg.data.thickness.smoothing_sd)

    
    return h,u_obs,sigma_x,sigma_y



@hydra.main(version_base=None, config_path="../configs", config_name="ekstrom_final")
def process_flowline(cfg):
    #Load mesh files
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.mesh.dim == 2:
        mesh,Q_c,V = load_mesh(cfg)
    elif cfg.mesh.dim == 3:
        mesh2d,Q_c,mesh,Q,V,W = load_mesh(cfg)
    log.info("Loaded mesh")
    plot_mesh(mesh)

    #Load data into functions
    mesh1d,Q_emb,V_emb,coordinates,dist,tangent,normal = load_flowline(cfg)
    log.info("Loaded flowline")
    h_obs,u_obs,ux_err,uy_err = load_fields(cfg,Q_c,V)
    log.info("Loaded Remote Sensing Data")
    
    bed = firedrake.Constant(-1.0e6)
    s0 = icepack.compute_surface(thickness=h_obs,bed=bed,rho_I = rho_ice,rho_W = rho_sw)
    b0 = firedrake.assemble(s0-h_obs)

    # Save figures to show data and mesh
    if cfg.save_figs:
        Path(workspace,"out",cfg.shelf,cfg.name).mkdir(parents=True, exist_ok=True)
        if cfg.mesh.dim == 2:
            fig,ax = plot_mesh(mesh)
        elif cfg.mesh.dim == 3:
            fig,ax = plot_mesh(mesh2d)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"mesh.png"))

        fig,ax = tripcolor(h_obs,title = "Initial thickness")
        plot_transect(coordinates,ax=ax)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"initial_thickness.png"))
        fig,ax = tripcolor(u_obs,title = "Initial velocity")
        #ip_streamplot(u0,title = "Final velocity",density = 1000,percision=10,ax=ax)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"initial_velocity.png"))
        fig,axes = plot_all_1d(coordinates,dist,[h_obs,u_obs,[s0,b0]],labels=["Thickness","Velocity","Profile"])
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"initial_1d.png"))

    #Create icepack model and solver
    model,diagnostic_solver_kwargs = icepack_model(cfg.icepack_model,Q_c,V)
    logging.info(diagnostic_solver_kwargs)
    log.info("Created model")
    opts = icepack_solver(cfg.flow_solver,model)
    solver = icepack.solvers.FlowSolver(model, **opts) 
    log.info("Created solver")

    u0 = solver.diagnostic_solve(
            velocity= u_obs, 
            thickness=h_obs, 
            surface = s0,
            **diagnostic_solver_kwargs,
        )

    #Define inverse problem
    inverse_param = cfg.inverse.get("param",None)
    if inverse_param is None:
        #If we want to take the data as it is, then just take the observed velocity
        u_estimate = u_obs
    elif inverse_param not in diagnostic_solver_kwargs.keys():
        raise ValueError(f"Parameter {inverse_param} not found in diagnostic_solver_kwargs")
    else:
        #Otherwise, create an inverse problem and solve it
        fwd_sim = create_fwd_model(inverse_param,solver,u_obs,h_obs,diagnostic_solver_kwargs)
        area = firedrake.Constant(firedrake.assemble(firedrake.Constant(1.0) * firedrake.dx(mesh)))
        loss = create_loss(cfg.inverse.loss,u_obs,h_obs,area)
        reg = create_reg(cfg.inverse.reg,u_obs,h_obs,area)
        log.info("Created inverse problem")
        problem = StatisticsProblem(
            simulation=fwd_sim,
            loss_functional=loss,
            regularization=reg,
            controls= firedrake.Function(Q_c)
        )
        estimator = MaximumProbabilityEstimator(
            problem,
            gradient_tolerance=cfg.inverse.optim.gradient_tolerance,
            step_tolerance=cfg.inverse.optim.step_tolerance,
            max_iterations=cfg.inverse.optim.max_iterations,
        )
        log.info("Created estimator")
        theta_estimate = estimator.solve()
        log.info("Estimated parameters")
        u_estimate = fwd_sim(theta_estimate)
        log.info("Estimated best velocity")


    if cfg.save_figs:
        fig,ax = tripcolor(h_obs,title = "Final thickness")
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"final_thickness.png"))
        fig,ax = tripcolor(u_estimate,title = "Final velocity")
        #ip_streamplot(u0,title = "Final velocity",density = 1000,percision=10,ax=ax)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"final_velocity.png"))

    #Output fluxes and save all variables
    dQxdx = firedrake.interpolate((h_obs*u_estimate).dx(0)[0],Q_c)
    dQydy = firedrake.interpolate((h_obs*u_estimate).dx(1)[1],Q_c)
    flux_divergence = firedrake.interpolate(firedrake.as_vector((dQxdx,dQydy)),V)
    tmb = firedrake.interpolate(dQxdx+dQydy,Q_c)
    smb = firedrake.interpolate(firedrake.Constant(0.3) ,Q_c)
    bmb = firedrake.interpolate(smb-tmb,Q_c)
    log.info("Defined fluxes")

    u_along = project_onto_dir(u_estimate.at(coordinates),tangent)
    u_across = project_onto_dir(u_estimate.at(coordinates),normal)
    dQ_along = project_onto_dir(flux_divergence.at(coordinates),tangent)
    dQ_across = project_onto_dir(flux_divergence.at(coordinates),normal)
    log.info("Projected fields")
    
    df = output_regular(coordinates,dist,cfg.flowline.out_npoints,s0,b0,u_along,dQ_along,dQ_across,smb,bmb,tmb)
    fname = Path(workspace,"out",cfg.shelf,cfg.name,"flowline.csv")
    if not Path(fname) or cfg.overwrite:
        df.to_csv(fname)


if __name__ == "__main__":
    process_flowline()