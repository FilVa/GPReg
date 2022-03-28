    # -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:10:58 2019

@author: filipavaldeira
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Plot shapes from list
def plot_dataset(shapes_list,title,legend) : 
    
    color= ['blue','red','green','orange','black','yellow']
    mark = ['o','x', 's','*','o','.','+']
    fig = plt.figure()    
    dim = shapes_list[0].shape[1]
    
    # 2D shapes
    if(dim==2):
        ax = fig.add_subplot(111)
        for i,shape in enumerate(shapes_list):
            ax.scatter(shape[:,0],shape[:,1],c=color[i],marker=mark[i],s=2, label = legend[i])
        
    # 3D shapes
    else:
        ax = fig.add_subplot(111, projection='3d')
        for i,shape in enumerate(shapes_list):
            zline = shape[:,2]
            xline = shape[:,0]
            yline = shape[:,1]            
            ax.scatter3D(xline, yline, zline,c=color[i], marker=mark[i], s=2,label = legend[i])
    
    # Set legend and title  
    ax.legend(frameon=False)
    ax.set(title = title)
    plt.show()
    return ax


def plot_with_ids(template,target = None, xlim=None):
    i=1
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes()
    ax.scatter(template[:,0],template[:,1],label = 'template',s=20,c='red')
    for vertex in template:
        xdata = vertex[0]
        ydata = vertex[1]
        txt = str(i)
        ax.text(xdata+0.01, ydata+0.01, txt)
        i +=1
    if target is not None:
        i=1
        ax.scatter(target[:,0],target[:,1],label = 'target',s=20,c='blue')
        for vertex in target:
            xdata = vertex[0]
            ydata = vertex[1]
            txt = str(i)
            ax.text(xdata+0.01, ydata+0.01, txt)
            i +=1
        plt.legend( loc='upper left')
    ax.set_aspect('equal')
    if xlim is not None:
        plt.xlim(xlim)

############################NOT TESTED YET



# Gets point matrix (n_points*dim*n_samples) and plots different samples with different colours
def plot_shapes(data, title, legend) :
   
    color= ['blue','red','green','orange','black']
    mark = ['o','x', 'o','*']
    fig = plt.figure()
    
    # 2D shapes
    if(data.shape[1]==2):        
        ax = fig.add_subplot(111)   
        if(len(data.shape)==2):
            ax.scatter(data[:,0],data[:,1],c=color[0],marker=mark[0], label = legend)
        else:            
            for i in range(data.shape[2]):
                ax.scatter(data[:,0,i],data[:,1,i],c=color[i],marker=mark[i], label = legend[i])
        ax.set(title = title)
    
    # 3D shapes
    else:
        ax = fig.add_subplot(111, projection='3d')
        
        if len(data.shape)==2:
            zline = data[:,2];
            xline = data[:,0];
            yline = data[:,1];
            ax.scatter3D(xline, yline, zline,c=color[0], label = legend,s=2)
        else :
            for i in range(data.shape[2]):
                zline = data[:,2,i];
                xline = data[:,0,i];
                yline = data[:,1,i];
                ax.scatter3D(xline, yline, zline,c=color[i],s=2,marker=mark[i], label = legend[i])
    
    # Set legend, etc            
    ax.legend()
    ax.set(title = title)
    #plt.show()
    return ax

# Plot two shapes, showing the connection between points with same index and plotting extra points which do not have connection
def plot_connected_shapes(shape1_conn, shape2_conn, title, legend, begin, end, shape1_extra = None,shape2_extra= None):
    fig = plt.figure()
    color= ['blue','red','dodgerblue','green']#change light red
    dim = shape1_conn.shape[1]    
    
    
    if(dim==3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(shape1_conn[:,0], shape1_conn[:,1], shape1_conn[:,2], label = legend[0], marker='o',c=color[0],s=5)
        ax.scatter3D(shape2_conn[:,0], shape2_conn[:,1], shape2_conn[:,2],c=color[1],marker='x', label = legend[1],s=5)
        
        if(shape1_extra is not None):
            ax.scatter3D(shape1_extra[:,0], shape1_extra[:,1], shape1_extra[:,2], marker='o',  c = color[2], label = legend[2],s=2)
        if(shape2_extra is not None):
            ax.scatter3D(shape2_extra[:,0], shape2_extra[:,1], shape2_extra[:,2],c=color[3], marker='x', label = legend[3],s=2)
        
        # Plot connections between data 1 and 2
        for i in range(begin,end) :
            ax.plot3D((shape1_conn[i,0],shape2_conn[i,0]), (shape1_conn[i,1],shape2_conn[i,1]),(float(shape1_conn[i,2]),shape2_conn[i,2]), c='black')
    
    # 2D shapes
    elif(dim==2):
        ax = fig.add_subplot(111)
        
        ax.scatter(shape1_conn[:,0], shape1_conn[:,1], s=80, marker='o', facecolors='none', edgecolors = color[0], label = legend[0])
        ax.scatter(shape2_conn[:,0], shape2_conn[:,1],c=color[1],marker='x', label = legend[1])
        
        if((shape1_extra is not None) and (shape1_extra.shape[1] != 0)):
            ax.scatter(shape1_extra[:,0], shape1_extra[:,1], s=80, marker='o', facecolors='none', edgecolors = color[2], label = legend[2])
        if((shape2_extra is not None) and(shape2_extra.shape[1] != 0)):
            ax.scatter(shape2_extra[:,0], shape2_extra[:,1],c=color[3],marker='x', label = legend[3])          
        
        # Plot connections between data 1 and 2
        for i in range(begin,end) :
            ax.plot((shape1_conn[i,0],shape2_conn[i,0]),(shape1_conn[i,1],shape2_conn[i,1]),c='black')
    
    #ax.set(title = title)
    ax.legend()
    plt.axis('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5),frameon=False)
    plt.show()
    return ax,fig



# PLOT GP RESULTS
def plot_deformations(template,target,points,noise,deform,vmin=0,vmax=0.1):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(template[:,0],template[:,1],label='template',c='black',s=20,marker ='*')
    ax.scatter(target[:,0],target[:,1],label='target',c='blue',s=20,marker ='*')
    im = ax.scatter(points[:,0],points[:,1],c =noise,s = 40,vmin=vmin,vmax=vmax)

    for i in np.arange(points.shape[0]):
        ax.arrow(points[i,0],points[i,1], deform[i,0],deform[i,1])

    #ax.quiver(points[:,0],points[:,1], deform[:,0],deform[:,1],noise)
    cbar =plt.colorbar(im)    
    cbar.set_label('Variance')
    plt.legend()
    return fig,ax