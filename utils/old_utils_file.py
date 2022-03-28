# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:03:39 2021

@author: filipavaldeira
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import scipy.io as sio
import numpy as np
from scipy.linalg import orthogonal_procrustes

# Plot 3d
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from utils_alignment import plotSample
from mpl_toolkits.mplot3d import Axes3D


import os
import pandas as pd
import trimesh
import shutil


# Rotate P into Q
# Return rotated P 'solution' and rotation matrix 'R'
def OPA(Q,P) :
      res =orthogonal_procrustes(P,Q);
      R = res[0]
      solution = np.matmul(P, R);
      return solution, R


# Cost function of GPA
def getG(Xp,m,k,n):
    
    Xps = np.reshape(Xp,(m*k,n));
    G = 0;
    for i in range(n) :
        rept = np.tile(Xps[:,i],(n-i-1,1));
        others = Xps[:,i+1:n];
        diff = rept - np.transpose(others);
        squared = np.square(diff);
        G = G+ np.sum(squared);
    G = G/n;   
    return G
    
# Generalized Procrustes Analysis         
# Receives k*m*n matrix 'shapes' : k is number of landmarks, m is dimension, n is number of samples
# Applies GPA with scaling if scale_flag =1
def GPA(shapes, scale_flag, thresh_tol):
    print('Start Generalized Procrustes Analysis')
    [n_points,dim,n_samples] = shapes.shape
    
    transformations = np.zeros((dim+1,dim+1,n_samples))
    X_bar= np.zeros((n_points,dim))
    X_p = np.zeros((n_points,dim,n_samples))
    
    # Translations
    print('Translation step')
    C= np.identity(n_points)-(1/n_points)*(np.ones((n_points,1))*np.ones((1,n_points))) 
    for i in range(n_samples):
        X_p[:,:,i] = np.matmul(C,shapes[:,:,i])
        diff =  X_p[:,:,i] - shapes[:,:,i]
        transformations[0:3,3,i] = diff[0,:]
                    
    #check_matrix = np.zeros((n_points,dim+1,n_samples));
    for i in range(n_samples):
        transformations[0:3,0:3,i] = np.identity(dim)
        transformations[3,3,i] = 1
        #check_matrix[:,:,i] = np.matmul(np.append(shapes[:,:,i],np.ones((n_points,1)),axis=1), np.transpose(transformations[:,:,i]))
    
    # Initialize tolerance
    tol_old = 1e16
    tol_new = 1e15
    while (abs(tol_old - tol_new) > thresh_tol):
        # Rotation cycle
        print('Rotation cycle')
        while (abs(tol_old - tol_new) > thresh_tol):            
            tol_old = tol_new;            
            for i in range(n_samples):                
                # exclude sample from mean
                before_sample = X_p[:,:,0:i]
                after_samples = X_p[:,:,i+1:n_samples]
                join =  np.zeros((n_points,dim,n_samples-1))
                join[:,:,0:i] = before_sample
                join[:,:,i:n_samples] = after_samples
                
                X_bar = np.mean(join,2) # Mean  shape
                X_p[:,:,i],rotation = OPA(X_bar, X_p[:,:,i])
                # Save rotation matrix
                transformations[0:3,0:3,i] = np.matmul(transformations[0:3,0:3,i],np.transpose(rotation)) 
                # Update translation
                transl_rot= np.matmul(np.transpose(transformations[0:3,3,i]),rotation)
                transformations[0:3,3,i] = np.transpose(transl_rot)                
           
            tol_new = getG(X_p,dim,n_points,n_samples);
        
        # Scaling
        print('Scaling')
        if(scale_flag ==1) : 
            beta = np.zeros((n_samples))
            vecXp = np.reshape(X_p,(n_points*dim, n_samples)) # Columns are observations, each row is a coordinate
            Phi = np.corrcoef(vecXp, rowvar=False);
            [eigval,eigvec] = np.linalg.eig(Phi)
            sorted_id = np.argsort(eigval)
            largest_eigvec = eigvec[:,sorted_id[-1]] # eigenvector of largest eigenvalue of corr matrix
            
            sumOfSquaredNorms = 0
            for i in range(n_samples):
                sumOfSquaredNorms = sumOfSquaredNorms + math.pow(np.linalg.norm(X_p[:,:,i]),2);
            for i in range(n_samples):
                norm_sqr = math.pow(np.linalg.norm(X_p[:,:,i]),2)
                beta[i] = math.sqrt(sumOfSquaredNorms/norm_sqr)*abs(largest_eigvec[i]);
                
                # Update transformation matrix
                transformations[:,:,i] =  transformations[:,:,i]*beta[i]
                X_p[:,:,i] = beta[i] * X_p[:,:,i];
            tol_new = getG(X_p,dim,n_points,n_samples)
    
    # Correct transformation matrix
    for i in range(n_samples):
        transformations[3,3,i] = 1
    X_bar = np.mean(X_p,2) # Final mean shape
    
    return transformations, X_bar


# Reads data and transformations from root and saves transformed data in destination
# Starts from ID start_id
def apply_transformation(root,destination,start_id) :

    for folder in os.scandir(root):
        id_subject = int(os.path.basename(folder.path))
        if folder.is_dir() and (id_subject >= start_id) :
            print("Checking folder " + folder.path) 
            # Reset flags
            found_alignment = False
            found_obj_data = False
            # Find object file with mesh and txt file with transformation matrix
            for file in os.listdir(folder):  
                if file.endswith(".obj"):
                    mesh_file = os.path.join(folder, file)
                    print("- Found obj file: " + mesh_file)
                    found_obj_data = True
                if file.endswith("alignment.txt"):
                    alignment_filename = file
                    alignment_file = os.path.join(folder, file)
                    print("- Found alignment txt file: " + alignment_filename)
                    found_alignment = True
            
            if not found_obj_data:
                print("- Missing .obj file with face scan")
                continue
            if not found_alignment:
                print("- Missing .txt alignment file with transformation matrix")
                continue
            
            print('- All files found : apply transformation')
            scan_mesh = trimesh.load(mesh_file, process=False) # Load obj file
            print("- Imported obj file")
            
            # Read transformation matrix into array
            alignment_descriptor_read = open(alignment_file, "r")
            alignment_matrix = np.array([[float(x) for x in line.strip().split(' ')] for line in alignment_descriptor_read])
            alignment_descriptor_read.close()
            # Apply transformation
            scan_mesh.apply_transform(alignment_matrix)
            
            # Save transformed obj file to destination
            if not os.path.exists(destination):
                os.makedirs(destination)
            print("- Saving data at", os.path.join(destination, os.path.basename(folder.path) + ".obj"))            
            scan_mesh.export(os.path.join(destination, os.path.basename(folder.path) + ".obj"))
            print('- Completed -')
            


# For files in root directory write txt with transformation matrix from transformations
# Only for files whose ID are in id_list
def write_transformations_txt(root,transformations,id_list):

    for folder in os.scandir(root):
        if folder.is_dir():        
            print("Checking folder " + folder.path)  
            id_subj = int(os.path.basename(folder.path))
            # Subject is in adquired transformations matrix
            if id_subj in id_list:
                print('Has transformation available : creating alignment.txt')
                destinationFolder = os.path.join(root, os.path.basename(folder.path))
                idx = np.where(id_list == id_subj)
                mat = transformations[:,:,idx[0][0]]; # get corresponding transf
                with open(os.path.join(destinationFolder, "alignment.txt"),'w') as f:
                    for line in mat:
                        np.savetxt(f, line, fmt='%.8f',newline =' ')
                        f.write("\n")


# Run files in directory root and concatenates transformation matrices
# Returns :
#   - landmark_list_data : k*m*n matrix (k is number of landmarks, m is dimension, n is number of samples)
#   - id_list : array with ids of retrieved landmarks
def get_landmark_list(root):
    
    flag_first = 1;
    for folder in os.scandir(root):
        
        if folder.is_dir():
            print("Checking folder " + folder.path)
            
            #Extract files
            found_ids = False
            for file in os.listdir(folder):  
                if file.find("face_points") >= 0 and file.endswith(".txt"):
                    id_filename = file
                    id_file = os.path.join(folder, file)
                    print("- Found txt file: " + id_file)
                    found_ids = True
            
            # No file with landmarks
            if not found_ids:
                print("- Missing .txt file for landmarks: skip folder")
                continue
            
            # Read landmarks
            with open(id_file,'r') as fp:
                landmark_array = np.zeros((1,3));
                landmark_list_sample =  np.zeros((1,3));
                ctn_line = 0;
                for line in fp:
                    landmark_str = line.strip().split(' ');
                    landmark_array[0,0] = landmark_str[0];
                    landmark_array[0,1] = landmark_str[1];
                    landmark_array[0,2] = landmark_str[2];
                    if(ctn_line == 0):
                        landmark_list_sample[0,:] = landmark_array
                        ctn_line = 1
                    else :
                        landmark_list_sample = np.append(landmark_list_sample,landmark_array, axis=0);
            
            # Create landmark_list_data array     
            if(flag_first==1) :
                landmark_list_data = np.zeros((landmark_list_sample.shape[0],landmark_list_sample.shape[1],1))
                landmark_list_data[:,:,0] = landmark_list_sample
                id_list = np.zeros((1,1))
                id_list[0] = int(os.path.basename(folder.path))
                flag_first = 0
                
            else :
                # Add sample to landmark_list_data array
                temp = np.zeros((landmark_list_sample.shape[0],landmark_list_sample.shape[1],1))
                temp[:,:,0] = landmark_list_sample
                landmark_list_data = np.append(landmark_list_data,temp, axis=2);
                id_list = np.append(id_list,int(os.path.basename(folder.path)))
        
    return landmark_list_data, id_list




# Search in folder root and retrieve landmark point
# Creates face_points.txt with the landmarks in the folder
def write_landmark_files(root):
    
    flag_first = 1;
    for folder in os.scandir(root):
        
        if folder.is_dir():
            print("Checking folder " + folder.path)
            
            #Extract files
            found_ids = False
            found_model = False
            found_landmarks = False
            for file in os.listdir(folder):  
                if file.endswith(".obj"):
                    mesh_filename = file
                    mesh_file = os.path.join(folder, file)
                    found_model = True
                if file.find("ldmks") >= 0 and file.endswith(".txt"):
                    id_filename = file
                    id_file = os.path.join(folder, file)
                    found_ids = True
                if file.find("face_points") >= 0 and file.endswith(".txt"):
                    found_landmarks = True
            
        if not found_model:
            print("- Missing .obj file with face scan")
            continue                
        if not found_ids:
            print("- Missing .txt file for face landmarks")
            continue                 
        if found_landmarks:
            print("- face_points.txt already exists : skip folder")
            continue
        
        # Load mesh
        obj_descriptor_read = open(mesh_file, "r")
        obj_pts = np.array([[float(x) for x in line[2:].strip().split(' ')] for line in obj_descriptor_read if line.startswith("v ")])
        obj_descriptor_read.close()
        # Get landmarks        
        file_descriptor_read = open(id_file, "r")                
        ids = [(int(x)-1) for x in file_descriptor_read]
        print("- Loaded " + str(len(ids)) + " face landmarks")
        file_descriptor_read.close() 
        landmark_points = obj_pts[ids, 0:3]        


        # Write face_points file
        destinationFolder = os.path.join(root, os.path.basename(folder.path))
        with open(os.path.join(destinationFolder, "face_points.txt"),'w') as f:
            for line in landmark_points:
                np.savetxt(f, line, fmt='%.8f',newline =' ')
                f.write("\n")

# Copies obj files from src_path whose name is in id_vec to dest_path
def filter_samples(id_vec,src_path,dest_path) :
    
    # Loop files in src_path
    for file in os.listdir(src_path):
        
        if file.endswith(".obj"):
            id_filename = file
            id_file = os.path.join(src_path, file)
            print(id_file)
            sep = id_filename.split('.')
            id_subj = int(sep[0])
            # If file belongs to list then copy
            if(id_subj in id_vec):
                print(id_subj)
                dest = os.path.join(dest_path, file)
                print(dest)
                # Copy file
                shutil.copy(id_file, dest)

def convert_obj2csv(src_path, dest_path) :

    # Loop files in src_path
    for file in os.listdir(src_path):
        mesh_file = os.path.join(src_path, file)
        if file.endswith('.obj'):                
            obj_descriptor_read = open(mesh_file, "r")
            obj_pts = np.array([[float(x) for x in line[2:].strip().split(' ')] for line in obj_descriptor_read if line.startswith("v ")])
            obj_descriptor_read.close()
        elif file.endswith('.ply'):
            lineList= list()
            with open(mesh_file) as input_data:
            # Skips text before the beginning of the interesting block:
                for line in input_data:
                    if line.strip() == 'end_header':  # Or whatever test is needed
                        break
                # Reads text until the end of the block:
                for line in input_data:  # This keeps reading the file
                    if len(line.strip().split(' ')) == 4:
                        break
                    lineList.append(line)
            obj_pts = np.array([[float(x) for x in ln.strip().split(' ')] for ln in lineList])
        
        sep = file.split('.')
        id_subj = sep[0]
        
        csv_path = os.path.join(dest_path, id_subj+'.csv')
        np.savetxt(csv_path, obj_pts, delimiter=",")
        
        print(csv_path)

def convert_csv2csv(src_path, dest_path, old_format) :

    # Loop files in src_path
    for file in os.listdir(src_path):
        if file.endswith(old_format):
            
            mesh_file = os.path.join(src_path, file)
            df = pd.read_csv(mesh_file,sep = ' ' ,header = None)
            
            sep = file.split('.')
            id_subj = sep[0]
            
            csv_path = os.path.join(dest_path, id_subj+'.csv')
            df.to_csv(csv_path, sep=',',index=False, header=False)
            
            print(csv_path)
                

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def noisy_copies(original_data_path, n_copies, std, dest_folder):
    
    df = pd.read_csv(original_data_path,header=None )
    original_points = df.to_numpy()
    plt.figure()
    ax = plt.axes()
    ax.scatter(original_points[:,0], original_points[:,1],c='k',marker='o')
    
    color = ['g','r','c','m','y','k','w']
    
    for cpy in range(2,n_copies+2):
        noisy_sample = np.random.normal(original_points, std)
        ax.scatter(noisy_sample[:,0], noisy_sample[:,1],c=color[cpy-2],marker='o')
        
        csv_path = os.path.join(dest_folder, str(cpy)+'.csv')
        df_noisy= pd.DataFrame(data=noisy_sample)
        #print(csv_path)
        df_noisy.to_csv(csv_path, sep=',',index=False, header=False)

