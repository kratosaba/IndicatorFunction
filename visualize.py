import numpy as np
import numpy.linalg as la
import mayavi.mlab as mlab
import meshcut

import common 


def points3d(verts, point_size=3, **kwargs):
    ax_scale = [1.0, 1.0, 1.0] # look into scale and change it to look normal
    if 'mode' not in kwargs:
        kwargs['mode'] = 'point'
    p = mlab.points3d(verts[:, 0], verts[:, 1], verts[:, 2], **kwargs)
    p.actor.property.point_size = point_size
    p.actor.actor.scale = ax_scale



def trimesh3d(verts, faces, **kwargs):

    ax_ranges = [-5, 5, -5, 5, -5, 5]
    ax_scale = [1.0, 1.0, 1.0] # look into scale and change it to look normal
    ax_extent = ax_ranges * np.repeat(ax_scale, 2)

    ax_scale = [1.0, 1.0, 1.0] # look into scale and change it to look normal

    surf = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces,
                         **kwargs)
    surf.actor.actor.scale = ax_scale
    
    mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])
    mlab.outline(surf, color=(1, 1, 1), extent=ax_extent)
    mlab.axes(surf, color=(1, 1, 1), extent=ax_extent,
              ranges=ax_ranges,
              xlabel='x', ylabel='y', zlabel='z')
    

def show(verts,faces,plane,show_mesh):
   
    mesh = meshcut.TriangleMesh(verts, faces)    
    P = meshcut.cross_section_mesh(mesh, plane)  
    if show_mesh:
        trimesh3d(mesh.verts, mesh.tris,color= (0.8,0.4,0.2),
                    opacity=1.0)

    for p in P:
        p = np.array(p)
        mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=None,
                    line_width=3.0, color=(0, 0, 1))
        
    return P
    
def vizualize_all_contours(verts,faces,scale,filename):

    equations = common.loadequation(filename)
    print(equations)
    i = 0
    show_mesh = True
    for equation in equations:
        if equation[0] == 0 and equation[1] != 0 and equation[2] != 0:
            x0 = 0
            y0 = (-equation[3] *scale)/ equation[1]
            z0 = (-equation[3] *scale)/ equation[2]
            # print(i)
        elif equation[0] == 0 and equation[1] == 0 and equation[2] != 0:
            x0 = 0
            y0 = 0
            z0 = (-equation[3] *scale)/ equation[2]
            # print(i)
        elif equation[0] == 0 and equation[1] != 0 and equation[2] == 0:
            x0 = 0
            y0 = (-equation[3] *scale)/ equation[1]
            z0 = 0
            # print(i)
        elif equation[0] != 0 and equation[1] == 0 and equation[2] != 0:
            x0 = (-equation[3] *scale) / equation[0]
            y0 = 0
            z0 = (-equation[3] *scale)/ equation[2]
            # print(i)
        elif equation[0] != 0 and equation[1] == 0 and equation[2] == 0:
            x0 = (-equation[3] *scale) / equation[0]
            y0 = 0
            z0 = 0
            # print(i)
        elif equation[0] != 0 and equation[1] != 0 and equation[2] == 0:
            x0 = (-equation[3] *scale) / equation[0]
            y0 = (-equation[3] *scale)/ equation[1]
            z0 = 0
            # print(i)
        else:    
            x0 = (-equation[3] *scale) / equation[0]
            y0 = (-equation[3] *scale)/ equation[1]
            z0 = (-equation[3] *scale)/ equation[2]
            # print(i)
        plane_orig = (x0, y0, z0)
        plane_norm = (equation[0], equation[1], equation[2])
        
        if i != 0:
            show_mesh = False
        plane = meshcut.Plane(plane_orig, plane_norm)
        P = show(verts,faces,plane,show_mesh)
        # print(len(P))
        i+=1

    mlab.show()
    
def vizualize_octree(octree,scale):

    xyz = octree.fullpointlist
    isovalue = octree.allisovalues
    mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2] ,colormap="copper", scale_factor=scale)
    mlab.show()
    

def vizualize_orgpos(orgpos):
    xyz = orgpos
    mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], colormap="copper", scale_factor=0.05)
    mlab.show()
