import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import icepack
import icepack.plot
import firedrake.plot
import scipy
from matplotlib.collections import LineCollection
import numpy as np
import logging

logger = logging.getLogger(__name__)

PATH = os.path.dirname(os.path.abspath(__file__))


color_opts = {
    "colors": {
        # "prior": [0.16078431, 0.26666667, 0.64705882],       #blue tueplots color for prior
        # "posterior": [1.0, 0.0, 0.0],    #red tueplots color for posterior
        # "observation": [0.49019608, 0.64705882, 0.29411765], #green tueplots color for observations/true values
        # "boundary_condition": [1.0, 171.0/255.0, 20.0/255.0] #orange tueplots color for boundary condition

        # "prior": "#d16800",       #orange blue prior
        # "posterior": "#7744d0",    #purple posterior
        # "observation": "#429400", #green for observations/true values
        # "boundary_condition": "#570041", #dark green for boundary conditions
        # "contrast1": "#c96940",
        # "contrast2": "#3a575a",

        # "prior": "#048ad1",       #light blue prior
        # "posterior": "#841ea4",    #dark purple posterior
        # "observation": "#df3337", #orange for observations/true values
        # "boundary_condition": "#4f8522", #dark green for boundary conditions
        # "contrast1": "#5310f0",
        # "contrast2": "#bd7135",

        "prior": "#069d3f",       #orange blue prior
        "posterior": "#140289",    #purple posterior
        "observation": "#a90505", #green for observations/true values
        "boundary_condition": "#825600", #dark green for boundary conditions
        "contrast1": "#808080",
        "contrast2": "#000000",
    },
    "color_maps":{
        "ice": mpl.cm.get_cmap("YlGnBu"),
        "age": mpl.cm.get_cmap("magma"),
        "prior_pairplot": mpl.cm.get_cmap("Blues"),
        "posterior_pairplot": mpl.cm.get_cmap("Reds"),
        "noise": mpl.cm.get_cmap("tab10")
    },
    "color_cycles":{
        "standard": plt.rcParams['axes.prop_cycle'].by_key()['color'],
    }
}

def setup_plots(**kwargs):
    style = kwargs.get("style","standard")
    plt.style.use(PATH + os.sep + style + ".mplstyle")
    return color_opts


def plot_mesh(mesh, **kwargs):
    """Plot a mesh using matplotlib."""
    fig, axes = icepack.plot.subplots()
    icepack.plot.triplot(mesh, axes=axes,boundary_kw={"linewidths":5.0})
    axes.legend()
    return fig, axes

def tripcolor(var,**kwargs):
    """Plot a scalar field with a colorbar."""
    fig,ax = plt.subplots()
    ax.tick_params(axis="both")
    title = kwargs.get("title",None)
    ax.set_title(title)
    ax.tick_params(axis="both")
    cmap = kwargs.get("cmap",None)
    colors = icepack.plot.tripcolor(var, axes=ax,cmap = "Blues")
    cbar = fig.colorbar(colors,ax=ax)
    cbar_label = kwargs.get("cbar_label",None)
    cbar.set_label(cbar_label)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels2 = [0]
    for label in labels[1:]:
        labels2.append(int(label)/1000)
    labels = labels2

    ax.set_xticklabels(labels)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels2 = [0]
    for label in labels[1:]:
        labels2.append(int(label)/1000)
    return fig,ax


def ip_streamplot(u, **kwargs):
    """Draw streamlines of a vector field. Adapted from icepack.plot.streamplot."""
    if u.ufl_shape != (2,):
        raise ValueError("Stream plots only defined for 2D vector fields!")

    u = icepack.plot._project_to_2d(u)
    axes = kwargs.get("ax",plt.gca())
    title = kwargs.get("title",None)
    axes.set_title(title)

    mesh = u.ufl_domain()
    coordinates = icepack.plot._get_coordinates(mesh)
    precision = kwargs.pop("precision", 10*icepack.plot._mesh_hmin(coordinates))
    density = kwargs.pop("density", 0.1 * icepack.plot._mesh_hmin(coordinates))
    max_num_points = kwargs.pop("max_num_points", np.inf)
    coords = coordinates.dat.data_ro
    max_speed = icepack.norm(u, norm_type="Linfty")
    linewidth = kwargs.pop("linewidth", 0.5)

    tree = scipy.spatial.KDTree(coords)
    indices = set(range(len(coords)))

    vmin = kwargs.get("vmin", 0)
    vmax = kwargs.get("vmax", max_speed)
    trajectories = []
    line_colors = []
    while indices:
        x0 = coords[indices.pop(), :]
        try:
            s = icepack.plot.streamline(u, x0, precision, max_num_points)
            for y in s:
                for index in tree.query_ball_point(y, density):
                    indices.discard(index)

            points = s.reshape(-1, 1, 2)
            trajectories.extend(np.hstack([points[:-1], points[1:]]))

            speeds = np.sqrt(np.sum(np.asarray(u.at(s, tolerance=1e-10)) ** 2, 1))
            colors = (speeds - vmin) / (vmax - vmin)
            #line_colors.extend(cmap(colors[:-1]))
            line_colors.extend(np.array([0.8,0.8,0.8,1.0]))

        except ValueError:
            pass

    line_collection = LineCollection(trajectories, colors="black",linewidths=linewidth*np.ones_like(colors))
    axes.add_collection(line_collection)
    axes.autoscale_view()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return axes

def plot_transect(coordinates,ax):
    ax.plot(coordinates[:,0],coordinates[:,1],color = "red",linestyle="dashed",linewidth=2.0)
    return ax


units = {"thickness": "[m]", "velocity": "[m/a]", "log fluidity": "" , "profile": "[m.a.s.l]", "mass balance" : "[m/a]"}
def plot_all_1d(coordinates,dist,current_vars,labels=["Thickness" , "Velocity", "Profile"],initial_vars = []):
    """
    Plot the values of firedrake Functions along a 1D transect of coordiantes

    Arguments:
    coordinates: coordinates of the transect
    dist: distance along the transect
    current_vars: list of firedrake Functions to plot
    labels: labels for the plots
    initial_vars (optional): list of firedrake Functions to plot as dashed lines (for comparison)
    """
    nvars = len(current_vars)
    fig, axes = plt.subplots(ncols = nvars)
    for i in range(nvars):
        axes[i].set_title(labels[i])
        axes[i].set_ylabel(labels[i].lower() + " " + units[labels[i].lower()])
        axes[i].set_xlabel("distance [m]")
        cv = current_vars[i]
        if not isinstance(cv,list):
            cv = [cv]
        for v in cv:
            vs = np.array(v.at(coordinates,tolerance=1e-5))
            if v.ufl_shape == ():
                axes[i].plot(dist,vs,color=color_opts["colors"]["prior"])
            else:
                axes[i].plot(dist,np.linalg.norm(vs,axis=1),color=color_opts["colors"]["prior"])

            try:
                v0 = initial_vars[i]
                v0s = np.array(initial_vars[i].at(coordinates,tolerance=1e-5))
                if v0.ufl_shape == ():
                    axes[i].plot(dist,v0s,color=color_opts["colors"]["posterior"],linestyle="dashed")
                else:
                    axes[i].plot(dist,np.linalg.norm(v0s,axis=1),color_opts["colors"]["posterior"],linestyle="dashed")
            except:
                continue


    if len(initial_vars)>0:
        axes[-1].legend(["reconstructed","observed"])
    else:
        axes[-1].legend(["reconstructed"])



    fig.tight_layout()
    return fig,axes

