# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:41:55 2019

@author: filipavaldeira
"""

#from builtins import super

from utils.plot_shapes import plot_shapes
import matplotlib.pyplot as plt

from utils.transformations import apply_transformations

import open3d as o3d



class Shape(object):
    
    def __init__(self):
        #super().__init__(*args, **kwargs)
        #print('Creating shape')
        self.has_points = 0 #TODO change
        self.has_triangles = 0
        self.file_name = 'NOT SET'
        
    # ------- Build shape methods -------
    def add_points(self,points):
        self.points = points
        self.has_points = 1
        
    def add_triangles(self,triangles):
        self.triangles = triangles
        self.has_triangles = 1
        
    # ------- Get shape -------
    def get_pts_by_id(self,ids):
        return self.points[ids,:]
    
    def get_n_points(self):
        sz = self.points.shape
        return sz[0]
    
        
    # Returns correspondent o3dmesh
    def get_o3d_mesh(self):
        if(self.has_triangles):
            o3d_pts =  o3d.utility.Vector3dVector(self.points)
            o3d_tri =  o3d.utility.Vector3iVector(self.triangles)
            o3d_mesh = o3d.geometry.TriangleMesh(o3d_pts,o3d_tri)
            return o3d_mesh        
        else:
            print("No triangles available. Cannot produce mesh.")
    
    # ------- Transform shape -------
    
    # translates entire shape so that id_origin points is now located at (0,0,0)
    def set_origin(self,id_origin):
        translation = self.points[id_origin,:]
        self.points = self.points-translation
        
    def set_origin_by_user_pt(self,point):        
        translation = point
        self.points = self.points-translation   
        
    def apply_rigid_transform(self,tform):
      # tform(as the output from procrustes function)
      #      a dict specifying the rotation, translation and scaling that
      #      maps X --> Y
      self.points = apply_transformations(tform,self.points)
        
    # ------- Plotting methods -------
    def plot_shape(self):
        title = 'Shape plot'
        legend = 'shape points'
        plot_shapes(self.points,title,legend)
        
    def plot_shape_circle_pts(self,ids):
        title = 'Shape plot'
        legend = 'shape points'
        ax = plot_shapes(self.points,title,legend)
        points_add = self.points[ids,:]
        ax.scatter3D(points_add[:,0],points_add[:,1],points_add[:,2],c='red',marker = '*',s=50,label = 'points')
            
        ax.legend()
        plt.show()
    