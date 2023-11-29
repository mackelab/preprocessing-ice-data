import hydra
import firedrake
import icepack
from icepack.constants import ice_density as rho_ice, water_density as rho_sw
from preproc.flow_models import icepack_model,icepack_solver,create_initial_conditions,create_mass_balance
from preproc.utils.proj import trunc,get_flowline,output_regular
from preproc.utils.plot import setup_plots,tripcolor,ip_streamplot,plot_mesh
from preproc.utils.misc import get_project_root
from omegaconf import OmegaConf
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
workspace = get_project_root()
def create_mesh(cfg):
    type = cfg.mesh.type
    if type == "rectangle":
        mesh = firedrake.RectangleMesh(nx = cfg.mesh.nx, ny = cfg.mesh.ny, Lx = cfg.mesh.Lx,Ly = cfg.mesh.Ly)
    else:
        raise NotImplementedError(f"Mesh type {type} not implemented")
    
    mesh_family = cfg.mesh.get("mesh_family","CG")
    mesh_degree = cfg.mesh.get("mesh_degree",2)
    if cfg.mesh.dim == 2:
        Q = firedrake.FunctionSpace(mesh,mesh_family,mesh_degree)
        V = firedrake.VectorFunctionSpace(mesh,mesh_family,mesh_degree)
        return mesh,Q,V
    elif cfg.mesh.dim == 3:
        vertical_mesh_family = cfg.mesh.get("vertical_mesh_family","GL")
        vertical_mesh_degree = cfg.mesh.get("vertical_mesh_degree",2)
        mesh3d = firedrake.ExtrudedMesh(mesh, layers = 1)
        Q = firedrake.FunctionSpace(mesh3d,family=mesh_family,degree=mesh_degree,vfamily = vertical_mesh_family,vdegree = vertical_mesh_degree)
        Q_c = firedrake.FunctionSpace(mesh3d,family=mesh_family,degree=mesh_degree,vfamily = "R",vdegree = 0)
        V = firedrake.VectorFunctionSpace(mesh3d,family=mesh_family,degree=mesh_degree,vfamily = vertical_mesh_family,vdegree = 0,dim=2)
        W = firedrake.FunctionSpace(mesh3d,family="DG",degree=1,vfamily = vertical_mesh_family,vdegree = vertical_mesh_degree)
        return mesh,Q_c,mesh3d,Q,V,W
def define_flowline(cfg):
    type = cfg.mesh.type
    if type == "rectangle":
        x = np.linspace(0.05*cfg.mesh.Lx,0.95*cfg.mesh.Ly,cfg.flowline.npoints)
        y = 0.5*cfg.mesh.Ly*np.ones_like(x)
    else:
        raise NotImplementedError(f"Mesh type {type} not implemented")
    coordinates,dist = get_flowline(x,y,**cfg.flowline.smoothing)
    if cfg.mesh.dim == 3:
        coordinates = np.array([(x,y,0.0) for x,y in zip(coordinates[:,0],coordinates[:,1])])
    
    return coordinates,dist



@hydra.main(version_base=None, config_path="../configs", config_name="synthetic_long_3d")
def shelf_flowline(cfg):


    color_opts = setup_plots()
    #Create mesh
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.mesh.dim == 2:
        mesh,Q_c,V = create_mesh(cfg)
    elif cfg.mesh.dim == 3:
        mesh2d,Q_c,mesh,Q,V,W = create_mesh(cfg)

    log.info("Created mesh")
    log.info(cfg.flowline.smoothing)

    #Save figure of mesh
    if cfg.save_figs:
        Path(workspace,"out",cfg.shelf,cfg.name).mkdir(parents=True, exist_ok=True)
        if cfg.mesh.dim == 2:
            fig,ax = plot_mesh(mesh)
        elif cfg.mesh.dim == 3:
            fig,ax = plot_mesh(mesh2d)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"mesh.png"))
    
    #Create icepack model and solver
    model,diagnostic_solver_kwargs = icepack_model(cfg.icepack_model,Q_c,V)
    logging.info(diagnostic_solver_kwargs)
    log.info("Created model")
    opts = icepack_solver(cfg.flow_solver,model)
    solver = icepack.solvers.FlowSolver(model, **opts) 
    log.info("Created solver")

    #Define initial conditions before spinup
    h0,u_guess = create_initial_conditions(cfg.ics,mesh,Q_c,V,cfg.mesh.Lx,cfg.mesh.Ly)
    log.info("Created initial conditions")
    bed = firedrake.Constant(-1.0e6)
    s0 = icepack.compute_surface(thickness=h0,bed=bed,rho_I = rho_ice, rho_W = rho_sw)
    b0 = firedrake.assemble(s0-h0)
 
    u0 = solver.diagnostic_solve(
            velocity= u_guess, 
            thickness=h0, 
            surface=s0,
            **diagnostic_solver_kwargs
        )
    log.info("Completed initial diagnostic solve")
    #Define ground truth mass balance
    mass_balance = create_mass_balance(cfg.mass_balance,mesh,Q_c,cfg.mesh.Lx,cfg.mesh.Ly)
    log.info("Created mass balance")
    u = u0.copy(deepcopy=True)
    h = h0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)

    if cfg.save_figs:
        fig,ax = tripcolor(h0,title = "Initial thickness")
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"initial_thickness.png"))
        fig,ax = tripcolor(u0,title = "Initial velocity")
        #ip_streamplot(u0,title = "Initial velocity",density = 1000,percision=10,ax=ax)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"initial_velocity.png"))


    #Spinup
    log.info("Simulating")
    for step in range(cfg.spinup.num_timesteps):
        h = solver.prognostic_solve(
            cfg.spinup.dt,
            thickness=h, 
            velocity=u,
            accumulation=mass_balance,
            surface=s,
            thickness_inflow=h0
         )

        s = icepack.compute_surface(thickness=h,bed=firedrake.Constant(bed),rho_I = rho_ice, rho_W = rho_sw)
        b = firedrake.assemble(s-h)


        
        u = solver.diagnostic_solve(
            velocity=u,
            thickness=h,
            surface = s,
            **diagnostic_solver_kwargs,
        )
    log.info("Completed spinup")
    if cfg.save_figs:
        fig,ax = tripcolor(h,title = "Final thickness")
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"final_thickness.png"))
        fig,ax = tripcolor(u0,title = None)
        ip_streamplot(u,title = None,density = 3000,percision=100,ax=ax,linewidth=0.3)
        ax.spines['bottom'].set_bounds(0,cfg.mesh.Lx)
        ax.hlines(cfg.mesh.Ly/2,cfg.mesh.Lx/10,9*cfg.mesh.Lx/10,linestyles="solid",color=color_opts["colors"]["observation"],linewidth=1)
        fig.savefig(Path(workspace,"out",cfg.shelf,cfg.name,"synthetic_setup.png"))


    #Calculate fluxes and output all variables
    dQxdx = firedrake.interpolate((h*u).dx(0)[0],Q_c)
    dQydy = firedrake.interpolate((h*u).dx(1)[1],Q_c)
    tmb = firedrake.interpolate(dQxdx+dQydy,Q_c)
    smb = firedrake.interpolate(firedrake.Constant(0.3) ,Q_c)
    bmb = firedrake.interpolate(smb-mass_balance,Q_c)

    coordinates,dist = define_flowline(cfg)
    log.info("Defined flowline")

    df = output_regular(coordinates,dist,cfg.flowline.out_npoints,s,b,u,dQxdx,dQydy,smb,bmb,tmb)
    fname = Path(workspace,"out",cfg.shelf,cfg.name,"flowline.csv")
    if not Path(fname) or cfg.overwrite:
        df.to_csv(fname)

if __name__ == "__main__":
    shelf_flowline()

