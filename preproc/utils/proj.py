import numpy as np
import torch
from scipy.interpolate import interp1d,UnivariateSpline
import pandas as pd
import logging

logger = logging.getLogger(__name__)



def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def regrid(x,y,xnew,kind="cubic"):
    """
    Interpolate a function defined on one grid to another grid living on a subspace of the original grid

    Arguments:
    x: original grid
    y: function defined on original grid
    xnew: new grid
    """
    if y is None:
        return None
    f =  interp1d(x,y,bounds_error=False,kind=kind,fill_value="extrapolate",copy=True)
    return f(xnew)

def get_flowline(x,y,smooth=False,smoothing_constant=0.05):
    """
    Get the firedrake-friendly format for the coordinates of a flowline from the x and y coordinates of the flowline.
    Also returns the distance along the flowline. Smooth the flowline if necessary using a smoothing spline.

    Arguments:
    x: x coordinates of the flowline
    y: y coordinates of the flowline
    smooth: whether to smooth the flowline
    smoothing_constant: constant for the smoothing spline
    """
    coordinates = np.array([(x,y) for x,y in zip(x,y)])
    coordinates = trunc(coordinates,0)
    dist = np.concatenate((np.array([0.0]),np.cumsum(np.sqrt((coordinates[1:,0]-coordinates[:-1,0])**2 + (coordinates[1:,1]-coordinates[:-1,1])**2))))
    npoints = dist.size
    if smooth:
        alpha = np.linspace(0, dist[-1],dist.size)
        splines = [UnivariateSpline(dist, coords, w = smoothing_constant*np.ones(coords.size),s=npoints) for coords in coordinates.T]
        interpolated_points = np.vstack([spl(alpha) for spl in splines]).T
        coordinates = np.array([(point[0],point[1]) for point in interpolated_points[::-1]])
        new_dist = np.cumsum( np.sqrt(np.sum( np.diff(interpolated_points, axis=0)**2, axis=1)))
        new_dist = np.insert(new_dist, 0, 0)
        return coordinates,new_dist
    return coordinates,dist


def project_onto_dir(vector,dir):
    return np.sum(vector*dir,axis=1)
    
def output_regular(coordinates,dist,nx,s,b,u_along,dQxdx,dQydy,smb,bmb,tmb):
    """
    Output a regular grid of the given variables along a given set of coordinates.

    Arguments:
    coordinates: coordinates of the flowline
    dist: distance along the flowline
    nx: number of points in the regular grid
    s: surface elevation
    b: bed elevation
    u_along: velocity along the flowline
    dQxdx: x-component of the flux divergence
    dQydy: y-component of the flux divergence
    smb: surface mass balance
    bmb: basal mass balance
    tmb: total mass balance
    """

    logger.info("inside a function!")
    #can change xs here easily by specifying coarser/finer nx (number of points)
    new_dist = np.linspace(dist[0],dist[-1],nx)
    bs = np.array(b.at(coordinates,tolerance=1e-5))
    ss = np.array(s.at(coordinates,tolerance=1e-5))
    try:
        us = np.array(u_along.at(coordinates,tolerance=1e-5))[:,0]
    except:
        us = u_along
    bmbs = np.array(bmb.at(coordinates,tolerance=1e-5))
    smbs = np.array(smb.at(coordinates,tolerance=1e-5))
    tmbs = np.array(tmb.at(coordinates,tolerance=1e-5))
    try:
        dQxdxs = np.array(dQxdx.at(coordinates,tolerance=1e-5))
        dQydys = np.array(dQydy.at(coordinates,tolerance=1e-5))
    except:
        dQxdxs = dQxdx
        dQydys = dQydy


    
    bs = regrid(dist,bs,new_dist)
    ss = regrid(dist,ss,new_dist)
    us = regrid(dist,us,new_dist)
    bmbs = regrid(dist,bmbs,new_dist)
    smbs = regrid(dist,smbs,new_dist)
    tmbs = regrid(dist,tmbs,new_dist)
    dQxdxs = regrid(dist,dQxdxs,new_dist)
    dQydys = regrid(dist,dQydys,new_dist)

    data = np.array([new_dist,bs,ss,us,bmbs,smbs,tmbs,dQxdxs,dQydys]).T
    df = pd.DataFrame(data,columns =["x_coord","base","surface","velocity","bmb","smb","tmb","dQxdx","dQydy"])

    return df