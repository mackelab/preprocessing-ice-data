import pygmsh
import numpy as np
from pathlib import Path
import subprocess

def create_flowtube_mesh(xleft,yleft,xright,yright,coords_dir,coords_name,lcar=3e2):
    """
    Create a mesh for a flowtube from a csv file of coordinates of the left and right (flowline) boundaries
    The other boundaries are straight lines between the first and last points of the left and right boundaries
    Outputs .geo and .msh file at same location and same name.
    Arguments:
    xleft: x coordinates of the left flowline boundary
    yleft: y coordinates of the left flowline boundary
    xright: x coordinates of the right flowline boundary
    yright: y coordinates of the right flowline boundary
    coords_dir: directory to save the mesh files
    coords_name: name of the mesh files
    lcar: characteristic length of the mesh elements
    """

    # Create the mesh points in order
    geom = pygmsh.built_in.Geometry()
    flowline_down_pts = []
    flowline_up_pts = []
    for x,y in zip(xleft,yleft):
        point = geom.add_point([x,y,0.0],lcar=lcar)
        flowline_down_pts.append(point)

    for x,y in zip(xright,yright):
        point = geom.add_point([x,y,0.0],lcar=lcar)
        flowline_up_pts.append(point)

    # Create the lines and surface of the domain
    lines = [
            geom.add_line(flowline_up_pts[0],flowline_down_pts[0]),
            geom.add_spline(flowline_down_pts),
            geom.add_line(flowline_down_pts[-1],flowline_up_pts[-1]),
            geom.add_spline(flowline_up_pts[::-1])
    ]

    line_loop = geom.add_line_loop(lines)
    plane_surface = geom.add_plane_surface(line_loop)

    physical_lines = [geom.add_physical(curve) for curve in lines]
    physical_surface = geom.add_physical(plane_surface)
    geo_fname = Path(coords_dir,coords_name.stem + '.geo')
    print(geo_fname)

    with open(geo_fname, 'w') as geo_file:
        geo_file.write(geom.get_code())

    subprocess.run(["gmsh", "-2", "-format", "msh2", "-o", Path(coords_dir,coords_name.stem + ".msh"), geo_fname])

def create_flowline_mesh(xdata,ydata,coords_dir,coords_name,lcar=1.0e2):
    """
    Create a mesh for a flowline from its coordinates
    Outputs .geo and .msh file at same location and same name.

    Arguments:
    xdata: x coordinates of the left flowline boundary
    ydata: y coordinates of the left flowline boundary
    lcar is the characteristic length of the mesh elements.
    """

    geom = pygmsh.built_in.Geometry()
    lcar = 1.0e2
    flowline_pts = []

    for x,y in zip(xdata,ydata):
        point = geom.add_point([x,y,0.0],lcar=lcar)
        flowline_pts.append(point)


    lines = [
            geom.add_spline(flowline_pts),
    ]

    line_loop = geom.add_line_loop(lines)
    physical_lines = [geom.add_physical(curve) for curve in lines]
    geo_fname = Path(coords_dir,coords_name.stem + '.geo')
    with open(geo_fname, 'w') as geo_file:
        geo_file.write(geom.get_code())

    subprocess.run(["gmsh", "-1", "-format", "msh2", "-o", Path(coords_dir,coords_name.stem + ".msh"), geo_fname])
