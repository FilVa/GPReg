# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:38:22 2019

@author: filipavaldeira
"""

import numpy as np
import os
import pandas as pd
import pickle
import xlrd, xlwt
from xlutils.copy import copy as copy_exel
import open3d as o3d
from scipy.io import loadmat
from utils.convert import check_matrix

import sys




##############################################################################
# Read and write specific datasets
##############################################################################

# reads data from dataset fish and chinese character
def read_fish_data(folder,kind='def',level=1,sample=1):
    # kind : 'def','noise','occlusion','outlier'
    # level : 1 to 5
    # sample : 1 to 100
    
    filename = 'save_fish_'+kind+'_'+str(level)+'_'+str(sample)
    full_path = os.path.join(folder,filename)
    print('Reading {} '.format(full_path))
    
    data = loadmat(full_path)
    template = data['x1']
    target = data['y2a']
    def_template_goal = data['y2']
    
    return template, target, def_template_goal

def read_metrics_fish(methods,levels,results_folder,parent_path):
    metrics_list = list() # will have len = len(methods)
    reg_metrics_list = dict()
    reg_dataset = dict()
    
    for met in methods:
        reg_metrics_list[met] = dict()
        reg_dataset[met] = dict()
        
        df_method = pd.DataFrame() # store all levels for emthod        
        for lev in levels:
            
            main_path = os.path.abspath(os.path.join(parent_path, 'Level_'+str(lev),met,results_folder))
            
            path = os.path.abspath(os.path.join(main_path,'df_metrics.csv'))
            df_metrics = pd.read_csv(path,index_col=0)
            
            # concatenate
            df_method = pd.concat([df_method,df_metrics])
            path = os.path.abspath(os.path.join(main_path,'reg_metric.pkl'))
            reg_metrics_list[met][lev] = load_object(path)
            path = os.path.abspath(os.path.join(main_path,'reg_dataset.pkl'))
            reg_dataset[met][lev] = load_object(path)
        
        df_method = df_method.set_index([levels])
        metrics_list.append(df_method)
    
    
    
    return metrics_list,reg_metrics_list, reg_dataset



##############################################################################
# General read and write
##############################################################################

# Write shapes to csv files
def write_data(data,id_vec,dest_path,dim):
    """ Write shapes in data to csv files in dest_path folder

        Parameters
        ----------
        data : list or 3d matrix
            collection of shapes to write
            
        id_vec : array
            vector of ids in same order as shapes in data
        
        dest_path : folder where to write

    """ 
    
    n_samples = len(id_vec)
    
    if(n_samples==1):
        data = data.reshape(-1,dim)
        id_ = id_vec[0]
        csv_path = os.path.join(dest_path, str(id_)+'.csv')
        np.savetxt(csv_path, data, delimiter=",")
        print('Saved ID {} in {}'.format(str(id_),csv_path))          
        
    else:
        if(not isinstance(data, list) and (len(data.shape)==2)):
            # if it is a flat matrix make it 3D
            data = check_matrix(data,dim,int(data.shape[1]/dim),n_samples,'3d')        
        # Iterate over shapes
        for n_sample in range(n_samples):    
            if(isinstance(data, list)):
                shape = data[n_sample]
            else:
                shape = data[:,:,n_sample]
            # Process
            id_ = id_vec[n_sample]    
            csv_path = os.path.join(dest_path, str(id_)+'.csv')
            np.savetxt(csv_path, shape, delimiter=",")
            print('Saved ID {} in {}'.format(str(id_),csv_path))


##############################################################################
# Read an write meshes
##############################################################################

def read_mesh(path, remove_degenrate_flag = False):
    """ Reads mesh file and returns vertices and faces, or NOne if reading failed

        Parameters
        ----------
        path : str
            path to mesh file
            
        remove_degenrate_flag : Bool
            if True removes duplicated vertices and degenerate triangles (default False)

        Returns
        ------
        vertices : array
            array of vertices or None if failed reading
        
        faces : array
            array of faces or None if failed reading
    """    
    mesh = o3d.io.read_triangle_mesh(path,print_progress =False)
    
    # when o3d fails to read file returns mesh with 0 points and 0 triangles
    if not mesh.has_vertices(): 
        print('read_mesh failed to open Mesh: check file path')
        vertices, faces = None, None
    
    else:    
        if(remove_degenrate_flag):
            mesh = mesh.remove_duplicated_vertices()
            mesh = mesh.remove_degenerate_triangles()
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
    return vertices, faces

def write_mesh(path, pts, faces =None, normals = None):
    """ Writes mesh file either from pts, pts and faces, or pts, faces and normals

        Parameters
        ----------
        path : str
            path to mesh file
            
        pts : numpy array
            vertices of mesh
        faces : numpy array, optional
            faces of mesh
        normals : numpy array
            normals of mesh
            
    """    
    if faces is None:
        #print(' No faces provided')
        mesh = pts_to_mesh(pts)
        mesh.compute_vertex_normals()
    elif normals is None:           
        #print('No normals provided')
        mesh  =o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(pts),triangles =o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
    else:
        print('Not implemented when we have normals')

    o3d.io.write_triangle_mesh(path, mesh)

def convert_files(src_path, src_type,dest_path, dest_type, separation=',') :

    if not ((dest_type=='.csv')|(dest_type=='.stl')|(dest_type=='.txt')|(dest_type == '.ply')|(dest_type == '.obj')):    
        print('Cannot convert to chosen file type')
        sys.exit()
        
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Loop files in src_path
    for file in os.listdir(src_path):
        shape_file = os.path.join(src_path, file)
        if file.endswith(src_type):
            if(src_type=='.csv') | (src_type=='.txt'):
                df = pd.read_csv(shape_file,header=None,sep=separation)
                pts = df.values 
                mesh = pts_to_mesh(pts)
            elif(src_type=='.stl') | (src_type=='.ply') | (src_type=='.obj'):
                mesh = o3d.io.read_triangle_mesh(shape_file,print_progress =False)
                pts = np.asarray(mesh.vertices)
            else:
                continue
        
        sep = file.split('.')
        id_subj = sep[0]
        
        dest_path_file = os.path.join(dest_path, id_subj+dest_type)
        
        if(dest_type == '.csv') | (dest_type=='.txt'):
            np.savetxt(dest_path_file, pts, delimiter=",")
        elif(dest_type == '.stl')| (dest_type == '.ply')|(dest_type == '.obj'):
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            if not mesh.has_triangle_normals():
                mesh.compute_triangle_normals()
            success = o3d.io.write_triangle_mesh(dest_path_file, mesh)
            if not success:
                print('ERROR IN MESH WRITING. EXITING.')
                sys.exit()
        else:
            continue

def pts_to_mesh(points, remove_degenrate_flag = False):
    """ Gets faces of mesh
        Parameters
        ----------
        points : numpy array
            vertices of shape        
            
        remove_degenrate_flag : Bool
            if True removes duplicated vertices and degenerate triangles (default False)

        Returns
        ------
        mesh : o3d Triangle Mesh

    """    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    
    # Compute mesh from pt cloud    
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    if avg_dist==0:
        # due to duplicated points the dist to nearest neighbour is always zero
        mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    else:
        radii = [ avg_dist,avg_dist*1.5]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector(radii) )
    
    if remove_degenrate_flag:
            mesh = mesh.remove_duplicated_vertices()
            mesh = mesh.remove_degenerate_triangles()
            
    return mesh    

##############################################################################
# Save and load objects
##############################################################################

# Saves object in filename can be retreaved with load_object
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
# Loads object in filename previously saved with save_object
def load_object(filename):
    with open(filename, 'rb') as input:  
        obj = pickle.load(input)
    
    return obj

# read required data for registration
def load_reg_data(data_folder, flag_template_mesh = False):

    path = os.path.abspath(os.path.join(data_folder,'dataset.pkl'))
    og_dataset = load_object(path)

    path = os.path.abspath(os.path.join(data_folder,'complete_dataset.pkl'))
    complete_dataset = load_object(path)
    
    if flag_template_mesh:
        path = os.path.abspath(os.path.join(data_folder,'template.ply'))
        vertices, faces = read_mesh(path)
        template = vertices
        return og_dataset, complete_dataset, template, faces
    else:
        path = os.path.abspath(os.path.join(data_folder,'template.npy'))
        template =  np.load(path)
    
    return og_dataset, complete_dataset, template

def save_reg_results(results_path, df_metrics, df_params, reg_metric, reg_dataset):
    print('Saving results at\n {}'.format(results_path))
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    path = os.path.abspath(os.path.join(results_path, 'df_metrics.csv'))
    df_metrics.to_csv(path)

    path = os.path.abspath(os.path.join(results_path, 'df_params.csv'))
    df_params.to_csv(path)

    path = os.path.abspath(os.path.join(results_path, 'reg_metric.pkl'))
    save_object(reg_metric, path)

    path = os.path.abspath(os.path.join(results_path, 'reg_dataset.pkl'))
    save_object(reg_dataset, path)



##############################################################################
# EXCEL functions
##############################################################################

def excel_get_last_row(file,sheet):
    """Gets number of last row filled in excel

        Parameters
        ----------
        file : str
            path to excel file
            
        sheet : int
            sheet number

        Returns
        ------
        n_rows : int
            index of last row
    """

    read_book = xlrd.open_workbook(file) #Make Readable Copy
    sheet_handle = read_book.sheet_by_index(sheet)
    n_rows = sheet_handle.nrows
    return n_rows

def write_excel(file, sheet, row, col, content):
    """Writes on given column and row of an excel file

        Parameters
        ----------
        file : str
            path to excel file
            
        sheet : int
            sheet number where to write
            
        row : int
            row number where to write

        col : int
            col number where to write
        
        content : string
            a string to write on excel

        Returns
        ------
        n_rows : int
            index of last row

    """
    
    read_book = xlrd.open_workbook(file) #Make Readable Copy
    write_book = copy_exel(read_book) #Make Writeable Copy    
    write_sheet1 = write_book.get_sheet(sheet)
    write_sheet1.write(row,col,content)    
    write_book.save(file)

# gets a list and writes it to the first row available in excel
# first col : first column to start writing
def write_excel_list(file_path,sheet,content_list,first_col):
    
    last_row = excel_get_last_row(file_path,sheet)    
    col = first_col
    
    for elem in content_list:                
        write_excel(file_path,sheet,last_row,col,elem)         
        col += 1