# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:18:47 2022

@author: filipavaldeira
"""

import numpy as np
import pandas as pd
import datetime

from utils.convert import flat2mat, convert_to_dict, correspondence_switch
from utils.plotting import truth_table_plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.io import write_excel,excel_get_last_row, write_mesh

from metrics.general import mean_dist_organized

import itertools
import os
import pandas as pd


def metrics_labels():
    
    labels = ['Miss_acc','Miss_recall','Miss_precision',
                     'Out_acc','Out_recall','Out_precision',
                     'Success', 'Frac_corr', 'Exact_corr',
                     'Err_target','Err_def_template','Err_og_template','Time_per_shape']
    return labels

def handle_comb(list_comb,param_label):
    # receives list of combinations and returns dataframe with parameters for each
    c =  list(itertools.product(*list_comb)) # list of all combinations
    n_comb = len(c)
    row_labels = ['C'+str(i) for i in np.arange(n_comb)]
    
    df = pd.DataFrame(data = np.array(c), columns=param_label, index = row_labels)
    
    return df, n_comb, row_labels


#TODO : remove ground_truth and obtained 
class RegMetrics(object):
    def __init__(self,og_dataset,reg_dataset,thresh_dist_failed_reg,complete_dataset=None, template_faces = None):
        """ Class to compute registration metrics

            Input
            ----------
            og_dataset : Dataset
                dataset before registration with all applied trnasformations (missing, etc)
                
            reg_dataset : RegDataset
                output from registration
            
            complete_dataset : Dataset
                dataset before applying missing data and outliers if available (all shapes should have == number of points as template)
            
            thresh_dist_failed_reg : float
                # Distance in og shape - for non nan in both gt and reg we sort and compute the disfference in distance in original target
                # if this is larger than some threshold we consider a failed registration

        """ 

        # ---- Datasets ---- #
        ground_truth_corr = og_dataset.corresp_dict # this is size of sample correspondence with target
        result_corr = reg_dataset.corr_vec_list   # this is size of template     
        self.reg_dataset = reg_dataset
        self.og_dataset = og_dataset        
        self.thresh_dist_failed_reg = thresh_dist_failed_reg
        self.og_template = self.reg_dataset.template
        self.complete_dataset = complete_dataset
        if template_faces is not None:
            self.template_faces = template_faces
        
        
        # ---- Compute correspondences ---- #
        # Both correspondences are dictionaries, and the vectors have size of template
        self.corr_gt = self.switch_gt_corr(ground_truth_corr)
        self.corr_reg = convert_to_dict(reg_dataset.id_vec,reg_dataset.corr_vec_list)
        # Both correspondences are dictionaries, and the correspondences are in the view of target shape
        self.corr_gt_target_view = ground_truth_corr
        self.corr_reg_target_view= self.switch_reg_corr(self.corr_reg)

        # shapes with failed reg are not considered for metrics
        self.failed_reg = dict() 
        # reasons for failed reg
        # - no non-nan correspondence in common  
        
        # ---- Dictionaries to keep the metrics ---- #
        
        # Number of equal correspondences - exact correpsondences in gt and reg
        self.n_correct_corr = dict() 
        # Total non nan correspondences existing in the ground truth
        self.n_total_corr_gt = dict()
        # Total non nan correspondences found in registration
        self.n_total_corr_found = dict()
        # Total number of ids with non nan correspondences in common between GT and found
        # i.e. we found correspondence regardless of which id 
        self.n_total_corr_common = dict()
        
        # Fractions
        # non nan correspondences found as fraction of ground truth ones
        self.n_total_corr_found_over_gt = dict()
        # number of correspondences in common as fraction of ground truth ones
        self.n_total_corr_common_over_gt = dict()
        # Correct correspondences as fraction of gt ones
        self.frac_correct_corr = dict()
        
        # Distances
        # we check ids in target for 
        # which there is non nan correspondence in BOTH gt and registered, 
        # we check the mean distance in the original target of the sorted shapes, 
        # this is a good indicator of registration, but does not account for outliers and missing
        self.mean_dist_in_target = dict()
                
        
        # Missing data metrics
        self.missing_pts_TP, self.missing_pts_TN, self.missing_pts_FP, self.missing_pts_FN = dict(), dict(), dict(), dict()
        self.missing_acc, self.missing_recall, self.missing_precision = dict(), dict(), dict()
        # Outliers metrics
        self.outlier_pts_TP, self.outlier_pts_TN, self.outlier_pts_FP, self.outlier_pts_FN = dict(), dict(), dict(), dict()
        self.outlier_acc, self.outlier_recall, self.outlier_precision = dict(), dict(), dict()

        # Distance between deformed template and ground truth
        if self.complete_dataset is not None:
            self.mean_dist_def_template = dict() # distance between deformed template and target
            self.mean_dist_og_template = dict()
            #self.mean_dist_og_data_to_template_rigid = dict()
        
        # computes all metrics above
        self.compute_metrics()

         # ---- Dataset metrics - average over full dataset ---- #
         
        # average of self.mean_dist_in_target (only considers not failed registration)
        self.dataset_mean_error_target = np.array(list(self.mean_dist_in_target.values())).mean()
        if self.complete_dataset is not None:
            self.dataset_mean_dist_def_template =  np.array(list(self.mean_dist_def_template.values())).mean()
            self.dataset_mean_dist_og_template =  np.array(list(self.mean_dist_og_template.values())).mean()
        
        # Fraction of non failed registration
        wrong_reg = np.array(list(self.failed_reg.values()))
        self.dataset_frac_success_reg = ((len(wrong_reg)-wrong_reg.sum())/len(wrong_reg))
        # Fraction - Number of correspondences found/number of ground truth correspondences
        self.dataset_mean_corr_found_over_gt = np.array(list(self.n_total_corr_found_over_gt.values())).mean()
        # Fraction of exact correspondences with respect to total gt non nan
        self.dataset_mean_exact_corr_over_gt = np.array(list(self.frac_correct_corr.values())).mean()
        
        
        self.dataset_out_acc = np.nanmean(np.array(list(self.outlier_acc.values())))
        self.dataset_out_recall = np.nanmean(np.array(list(self.outlier_recall.values())))
        self.dataset_out_precision = np.nanmean(np.array(list(self.outlier_precision.values())))
        
        self.dataset_miss_acc = np.nanmean(np.array(list(self.missing_acc.values())))
        self.dataset_miss_recall = np.nanmean(np.array(list(self.missing_recall.values())))
        self.dataset_miss_precision = np.nanmean(np.array(list(self.missing_precision.values())))

    
    def compute_metrics(self):
        # cycle over each shape and computes all individual metrics
        
        # this replaces old results_corr_vec()        
        template = self.reg_dataset.template

        for id_ in self.reg_dataset.id_vec:
            og_shape = self.og_dataset.shapes_dict[id_].points
            def_template = self.reg_dataset.def_src_dict[id_]
            
            vec_gt = self.corr_gt[id_]
            vec_reg = self.corr_reg[int(id_)].flatten()
            equal_mask = np.equal(vec_gt,vec_reg)            
            mask_non_na =(~np.isnan(vec_gt)) & (~np.isnan(vec_reg))                      
            
            # ---- Detect possible failures in registration ---- #
            

            # TYPE 1 : there is no non-nan correspondence in common 
            if mask_non_na.sum() == 0:
                self.failed_reg[id_] = 1
                print('Shape {} has no non-nan correspondence - failed registration'.format(id_))
                continue 
            
            # TYPE 2 : MEan dist in original shape
            # Distance in og shape - for non nan in both gt and reg we sort and compute the disfference in distance in original target
            # if this is larger than some threshold we consider a failed registration
            #TODO fix
            diff = og_shape[vec_gt[mask_non_na].astype('int'),:]-og_shape[vec_reg[mask_non_na].astype('int'),:]
            dist = np.sqrt(np.sum(np.square(diff),axis=1))
            mean_dist = np.nanmean(dist)
              
            # # Detect failed registration
            if mean_dist > self.thresh_dist_failed_reg:
                self.failed_reg[id_] = 1
                print('Shape {} has mean_distance in targte corr larger than thresh_dist_failed_reg - failed registration'.format(id_))                
                continue
           
            # ---- Compute metrics only for successful registration ---- #
            
            self.failed_reg[id_]= 0 # registration did not fail
            self.mean_dist_in_target[int(id_)] = mean_dist  
            
            # Update dictionaries
            self.n_correct_corr[int(id_)] = np.sum(equal_mask)
            self.n_total_corr_gt[int(id_)] = np.sum(~np.isnan(vec_gt))
            self.n_total_corr_found[int(id_)] = np.sum(~np.isnan(vec_reg))
            self.n_total_corr_common[int(id_)] = np.sum(mask_non_na)
            
            self.n_total_corr_found_over_gt[int(id_)] = self.n_total_corr_found[int(id_)]/self.n_total_corr_gt[int(id_)]
            self.n_total_corr_common_over_gt[int(id_)] = self.n_total_corr_common[int(id_)]/self.n_total_corr_gt[int(id_)]
            self.frac_correct_corr[int(id_)] = self.n_correct_corr[int(id_)]/self.n_total_corr_gt[int(id_)]
            
            
            # Missing_pts
            TP, TN, FP, FN, accuracy, recall, precision  = self.truth_table_data(vec_gt,vec_reg)
            self.missing_pts_TP[int(id_)], self.missing_pts_TN[int(id_)], self.missing_pts_FP[int(id_)], self.missing_pts_FN[int(id_)]  = TP, TN, FP, FN,
            self.missing_acc[int(id_)], self.missing_recall[int(id_)], self.missing_precision[int(id_)] = accuracy, recall, precision
            
            # Outliers
            vec_gt_target, vec_reg_target = self.corr_gt_target_view[id_], self.corr_reg_target_view[id_]
            TP, TN, FP, FN, accuracy, recall, precision  = self.truth_table_data(vec_gt_target,vec_reg_target)
            self.outlier_pts_TP[int(id_)], self.outlier_pts_TN[int(id_)], self.outlier_pts_FP[int(id_)], self.outlier_pts_FN[int(id_)] = TP, TN, FP, FN
            self.outlier_acc[int(id_)], self.outlier_recall[int(id_)], self.outlier_precision[int(id_)] = accuracy, recall, precision
            

            
            
            

            
            # Distance metrics
            # distance to deformed temlate
            if self.complete_dataset is not None:
                full_shape = self.complete_dataset.shapes_dict[id_].points
                
                mean_err, max_err, min_err=  mean_dist_organized(def_template,full_shape)
                self.mean_dist_def_template[id_] = mean_err
                # Distance to original template
                mean_err, max_err, min_err=  mean_dist_organized(self.og_template,full_shape)
                self.mean_dist_og_template[id_] = mean_err
                # add here distance ot other like template rigid
           
                
    def truth_table_data(self,vec_gt,vec_reg):
        #computes true positive, etc and acc, prec, recall
        
        n_true = np.sum(np.isnan(vec_gt))
        n_found = np.sum(np.isnan(vec_reg))
        
        TP = np.sum(np.isnan(vec_reg)[np.isnan(vec_gt)])
        TN = np.sum(~np.isnan(vec_reg)[~np.isnan(vec_gt)])
        FP = np.sum(np.isnan(vec_reg)[~np.isnan(vec_gt)])
        FN = np.sum(~np.isnan(vec_reg)[np.isnan(vec_gt)])
        
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        
        
        if((TP+FN)!=0):
            recall = TP/(TP+FN)
        else :
            recall = np.nan
        if((TP+FP)!=0):
            precision = TP/(TP+FP)
        elif n_true == 0:
            precision = 1                    
        else:
            precision = np.nan
        return TP, TN, FP, FN, accuracy, recall, precision  
       
    def get_correct_template(self):
        #TODO - what is this...
        pt1_id = 4578 # top ear
        pt2_id = 507
        pt1_id_data = 5308
        pt2_id_data = 1432
        
        # True template
        template_path = r'C:/Users/filipavaldeira/Documents/Data_ears/Template/mean_shape.csv'            
        df = pd.read_csv(template_path,header=None)
        template = df.values
        pt1_true = template[pt1_id,:]
        pt2_true = template[pt2_id,:]
        print(pt1_true)
        print(pt2_true)
        
        fig = plt.figure()   
        ax = fig.add_subplot(111, projection='3d')
        zline = template[:,2]
        xline = template[:,0]
        yline = template[:,1]            
        ax.scatter3D(xline, yline, zline,c='blue', marker='o', s=2,label = 'template')
        ax.scatter3D(pt1_true[0], pt1_true[1], pt1_true[2],c='red', marker='x', s=30,label = 'template')
        ax.scatter3D(pt2_true[0], pt2_true[1], pt2_true[2],c='orange', marker='x', s=30,label = 'template')
        
        ax.legend()
        plt.show()
       
        
        for id_ in self.corr_gt.keys():
            # get first point, can be assigned or not, we don't knows
            if( ~ np.isnan(self.reg_dataset.def_template_non_ass[id_][pt1_id_data,:]).any()):
                pt1 = self.reg_dataset.def_template_non_ass[id_][pt1_id_data,:]
            else:
                pt1 = self.reg_dataset.def_template_ass[id_][pt1_id_data,:]
                
            # get second point
            if( ~ np.isnan(self.reg_dataset.def_template_non_ass[id_][pt2_id_data,:]).any()):
                pt2 = self.reg_dataset.def_template_non_ass[id_][pt2_id_data,:]
            else:
                pt2 = self.reg_dataset.def_template_ass[id_][pt2_id_data,:]

            fig = plt.figure()   
            ax = fig.add_subplot(111, projection='3d')
            zline = self.reg_dataset.def_template_non_ass[id_][:,2]
            xline = self.reg_dataset.def_template_non_ass[id_][:,0]
            yline = self.reg_dataset.def_template_non_ass[id_][:,1]            
            ax.scatter3D(xline, yline, zline,c='blue', marker='o', s=2,label = 'template')
            ax.scatter3D(pt1[0], pt1[1], pt1[2],c='red', marker='x', s=30,label = 'template')
            ax.scatter3D(pt2[0], pt2[1], pt2[2],c='orange', marker='x', s=30,label = 'template')
            
            ax.legend()
            plt.show()

                
            diff = pt1 -pt2
            diff_true = pt1_true - pt2_true
            
            dist = np.sqrt(np.square(diff).sum(axis=0))
            dist_true = np.sqrt(np.square(diff_true).sum(axis=0))
            
            
            print(dist)
            print(dist_true)
                

        

    

    ###########################################################################
    # PLOTS
    ###########################################################################
    def plot_dict_bar(self,axs,keys,values):
        axs.bar(keys,values)
        axs.set_xticks(list(keys))
        xlabel_ticks = [str(x) for x in keys]
        axs.set_xticklabels(xlabel_ticks)                
    
    
    
    def plot_dist_metrics_by_shape(self):
        if self.complete_dataset is not None:

            fig, axs = plt.subplots(3, 1, figsize=(10,10),sharex=True)
            
            
            self.plot_dict_bar(axs[0],self.mean_dist_in_target.keys(),self.mean_dist_in_target.values())
            axs[0].set_ylabel('Error distance')
            axs[0].set_title('Mean distance in target between true corr and pred')
            self.plot_dict_bar(axs[1],self.mean_dist_def_template.keys(),self.mean_dist_def_template.values())
            axs[1].set_ylabel('Error distance')
            axs[1].set_title('Mean distance between deformed template and target without missing/outliers')
            self.plot_dict_bar(axs[2],self.mean_dist_og_template.keys(),self.mean_dist_og_template.values())
            axs[2].set_ylabel('Error distance')
            axs[2].set_xlabel('Shape ids')
            axs[2].set_title('Mean distance between original template and target without missing/outliers')
    
            fig.suptitle('Distance metrics by shape')
            
        else:
            fig, axs = plt.subplots(1, 1, figsize=(10,10),sharex=True)
            
            self.plot_dict_bar(axs,self.mean_dist_in_target.keys(),self.mean_dist_in_target.values())
            axs.set_ylabel('Error distance')
            fig.suptitle('Mean distance in target between true corr and pred')
            
        
    def plot_corr_metrics_by_shape(self):
        fig, axs = plt.subplots(3, 1, figsize=(10,10),sharex=True)
        
        
        self.plot_dict_bar(axs[0],self.n_total_corr_found_over_gt.keys(),self.n_total_corr_found_over_gt.values())
        axs[0].set_ylabel('Ratio')
        axs[0].set_title('Total non-nan correspondences found (as fraction of gt ones)')
        self.plot_dict_bar(axs[1],self.n_total_corr_common_over_gt.keys(),self.n_total_corr_common_over_gt.values())
        axs[1].set_ylabel('Ratio')
        axs[1].set_title('Total non-nan correspondences in common with gt (as fraction of gt ones)')
        self.plot_dict_bar(axs[2],self.frac_correct_corr.keys(),self.frac_correct_corr.values())
        axs[2].set_ylabel('Ratio')
        axs[2].set_xlabel('Shape ids')
        axs[2].set_title('Exactly correct non-nan correspondences found (as fraction of gt ones)')

        fig.suptitle('Correspondence metrics by shape')
        
    
    def plot_outlier_metric_by_shape(self):
        fig, axs = plt.subplots(3, 1, figsize=(10,10),sharex=True)
        
        
        self.plot_dict_bar(axs[0],self.outlier_acc.keys(),self.outlier_acc.values())
        axs[0].set_ylabel('Accuracy')
        self.plot_dict_bar(axs[1],self.outlier_recall.keys(),self.outlier_recall.values())
        axs[1].set_ylabel('Recall')
        self.plot_dict_bar(axs[2],self.outlier_precision.keys(),self.outlier_precision.values())
        axs[2].set_ylabel('Precision')
        axs[2].set_xlabel('Shape ids')
        
        fig.suptitle('Outliers metrics by shape')
        
    
    def plot_missing_metric_by_shape(self):
        fig, axs = plt.subplots(3, 1, figsize=(10,10),sharex=True)
        
        
        self.plot_dict_bar(axs[0],self.missing_acc.keys(),self.missing_acc.values())
        axs[0].set_ylabel('Accuracy')
        self.plot_dict_bar(axs[1],self.missing_recall.keys(),self.missing_recall.values())
        axs[1].set_ylabel('Recall')
        self.plot_dict_bar(axs[2],self.missing_precision.keys(),self.missing_precision.values())
        axs[2].set_ylabel('Precision')
        axs[2].set_xlabel('Shape ids')
        
        fig.suptitle('Missing points metrics by shape')
    
    
    def plot_missing_by_shape(self):
        
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        
        axs[0,0].bar(self.missing_pts_TN.keys(),self.missing_pts_TN.values())
        axs[0,0].set_title('TN: Classified as non missing \n and it is non missing')
        axs[0,0].set_ylabel('Number of points')

        axs[0,1].bar(self.missing_pts_FN.keys(),self.missing_pts_FN.values())
        axs[0,1].set_title('FN: Classified as non missing \n but it was actually missing')
        axs[0,1].set_ylabel('Number of points')
        
        axs[1,0].bar(self.missing_pts_TP.keys(),self.missing_pts_TP.values())
        axs[1,0].set_title('TP: Classified as missing \n and it was missing')
        axs[1,0].set_ylabel('Number of points')

        axs[1,1].bar(self.missing_pts_FP.keys(),self.missing_pts_FP.values())
        axs[1,1].set_title('TP: Classified as missing \n but it was not missing')
        axs[1,1].set_ylabel('Number of points')
                
    def plot_outliers_by_shape(self):
        
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        
        axs[0,0].bar(self.outlier_pts_TN.keys(),self.outlier_pts_TN.values())
        axs[0,0].set_title('TN: Classified as non outlier \n and it is non outlier')
        axs[0,0].set_ylabel('Number of points')

        axs[0,1].bar(self.outlier_pts_FN.keys(),self.outlier_pts_FN.values())
        axs[0,1].set_title('FN: Classified as non outlier \n but it was actually outlier')
        axs[0,1].set_ylabel('Number of points')
        
        axs[1,0].bar(self.outlier_pts_TP.keys(),self.outlier_pts_TP.values())
        axs[1,0].set_title('TP: Classified as outlier \n and it was outlier')
        axs[1,0].set_ylabel('Number of points')

        axs[1,1].bar(self.outlier_pts_FP.keys(),self.outlier_pts_FP.values())
        axs[1,1].set_title('FP: Classified as outlier \n but it was not outlier')
        axs[1,1].set_ylabel('Number of points')
        
    def plot_truth_table_missing(self):
        
        mean_TP = np.array(list(self.missing_pts_TP.values())).mean()
        mean_TN = np.array(list(self.missing_pts_TN.values())).mean()
        mean_FP = np.array(list(self.missing_pts_FP.values())).mean()
        mean_FN = np.array(list(self.missing_pts_FN.values())).mean()        
        
        mean_acc = self.dataset_miss_acc
        mean_recall = self.dataset_miss_recall
        mean_precision = self.dataset_miss_precision
        
        cols = ['GT missing', 'GT not missing']
        rows = ['Reg missing', 'Reg not missing']
        title = ' Missing data truth table'           
        
        data_tab = [mean_TP, mean_TN, mean_FP, mean_FN]
        data_metrics = [mean_acc, mean_recall, mean_precision]
        truth_table_plot(data_tab,data_metrics,title,cols,rows)
        
        
    def plot_truth_table_outliers(self):
        mean_TP = np.array(list(self.outlier_pts_TP.values())).mean()
        mean_TN = np.array(list(self.outlier_pts_TN.values())).mean()
        mean_FP = np.array(list(self.outlier_pts_FP.values())).mean()
        mean_FN = np.array(list(self.outlier_pts_FN.values())).mean()        
        
        mean_acc = self.dataset_out_acc
        mean_recall = self.dataset_out_recall
        mean_precision = self.dataset_out_precision
        
        cols = ['GT outliers', 'GT not outliers']
        rows = ['Reg outliers', 'Reg not outliers']        
        title = 'Outliers truth table'        
        
        data_tab = [mean_TP, mean_TN, mean_FP, mean_FN]
        data_metrics = [mean_acc, mean_recall, mean_precision]
        truth_table_plot(data_tab,data_metrics,title,cols,rows)       

    
    def plot_correct_corresp_by_shape(self):
        fig, axs = plt.subplots(2, 1, figsize=(10,10))
        
        axs[0].bar(self.frac_correct_corr.keys(),self.frac_correct_corr.values())
        axs[0].set_title('Correct correspondences as fraction of gt ones')

        axs[1].bar(self.mean_dist_in_target.keys(),self.mean_dist_in_target.values())
        axs[1].set_title('Distance between corresponent points in the original target shape')        

    ###########################################################################
    # WRITE RESULTS
    ###########################################################################
    def dataset_metrics_summary(self, plot_flag = False):

        if plot_flag:
        
            fig, axs = plt.subplots(3, 1, figsize=(10,15))
            
            cols = ['Accurary','Recall','Precision']
            rows = ['Missing','Outliers']        
            data_miss_out = [[ self.dataset_miss_acc,self.dataset_miss_recall, self.dataset_miss_precision],
                    [ self.dataset_out_acc,self.dataset_out_recall, self.dataset_out_precision]]

            df = pd.DataFrame(data_miss_out,columns=cols, index = rows)
            df.plot.barh(ax=axs[0])
            axs[0].set_frame_on(False)
            axs[0].legend(frameon=False)
            axs[0].set_xlim([0,1])
            axs[0].set_title('Outliers and missing data metrics over dataset')
            
            # Correspondence metrics
            rows = ['Registration success','Total corr found/total gt corr','Exact corr found/total gt corr']     
            data_fractions = [[ self.dataset_frac_success_reg],
                [ self.dataset_mean_corr_found_over_gt],
                [self.dataset_mean_exact_corr_over_gt ]]  
            df = pd.DataFrame(data_fractions, index = rows)
            df.plot.barh(ax=axs[1])
            axs[1].set_xlim([0,1])
            axs[1].set_frame_on(False)
            axs[1].get_legend().remove()
            axs[1].set_title('Fraction metrics')

            # Distance errors
            # Correspondence metrics
            if self.complete_dataset is not None:
                rows = ['In target shape','Deformed template','Original template']     

            else:
                rows = ['In target shape']     
            if self.complete_dataset is not None:
                data_dist = [[ self.dataset_mean_error_target],
                        [ self.dataset_mean_dist_def_template],
                        [self.dataset_mean_dist_og_template ]]
            else:
                data_dist = [[ self.dataset_mean_error_target]]    
            df = pd.DataFrame(data_dist, index = rows)
            df.plot.barh(ax=axs[2])
    
            axs[2].set_frame_on(False)
            axs[2].get_legend().remove()
            axs[2].set_xlabel('Distance error')
            axs[2].set_title('Distance metrics')
            #TODO put metrics here maybe
        else:
            fig,axs= None, None

        metrics = [ self.dataset_miss_acc,self.dataset_miss_recall, self.dataset_miss_precision,
                   self.dataset_out_acc,self.dataset_out_recall, self.dataset_out_precision,
                   self.dataset_frac_success_reg, self.dataset_mean_corr_found_over_gt, 
                   self.dataset_mean_exact_corr_over_gt,self.dataset_mean_error_target,
                   self.dataset_mean_dist_def_template, self.dataset_mean_dist_og_template,self.reg_dataset.reg_time/self.reg_dataset.n_obs ]
        #TODO if this changes, change metrics_label function
        metrics_lab = metrics_labels()
        df_metrics = pd.DataFrame([metrics],columns=metrics_lab)

        
        return df_metrics,fig,axs
    
    def write_deformed_template(self,id_vec,folder_path):
        
        faces = self.template_faces
        
        for id_ in id_vec:
            def_template = self.reg_dataset.def_src_dict[id_]
            
            path = os.path.abspath(os.path.join(folder_path, 'Shape_'+str(id_)+'.ply'))
            write_mesh(path, pts = def_template, faces =faces)
        
        
    
    def get_standard_metrics(self):
        # PRints important metrics for overall dataset        
        print('Where we have correspondence in both sides: mean error distance in original shape')
        print(self.dataset_mean_error_target)

        print('Fraction of correct registration')
        print(self.dataset_frac_success_reg)
        
        print('Number of correspondences found/number of ground truth correspondences')
        print(self.dataset_mean_corr_found_over_gt)
        
        print('Fraction of exact correspondences with respect to total gt non nan')
        print(self.dataset_mean_exact_corr_over_gt)       
         
        return self.dataset_mean_error_target,self.dataset_mean_corr_found_over_gt,self.dataset_mean_exact_corr_over_gt
    
    
    def write_results_excel(self,file_path,sheet,comments,unique_key):
        
        last_row = excel_get_last_row(file_path,sheet)
        
        col = 1
        content = self.reg_dataset.reg_method
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 2
        content = self.result_frac_corr
        write_excel(file_path,sheet,last_row,col,content)

        col = 3
        content = self.result_mean_error_target
        write_excel(file_path,sheet,last_row,col,content)

        col = 4
        content = self.result_fraction_right
        write_excel(file_path,sheet,last_row,col,content)

        col = 5
        content = self.result_mean_error_target_correct
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 6
        content = self.reg_dataset.reg_time/self.reg_dataset.n_obs
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 7
        content = self.reg_dataset.parameters_str
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 8
        content = str(datetime.datetime.now())
        write_excel(file_path,sheet,last_row,col,content)
        
        # WRITE OUTLIERS
        self.mean_truth_table_outliers_pts(plot_flag=False)
        mean_TP = np.array(list(self.outlier_pts_TP.values())).mean()
        mean_TN = np.array(list(self.outlier_pts_TN.values())).mean()
        mean_FP = np.array(list(self.outlier_pts_FP.values())).mean()
        mean_FN = np.array(list(self.outlier_pts_FN.values())).mean()

        col = 9
        content = mean_TP
        write_excel(file_path,sheet,last_row,col,content)

        col = 10
        content = mean_TN
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 11
        content = mean_FP
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 12
        content = mean_FN
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 13
        content = self.out_accuracy
        write_excel(file_path,sheet,last_row,col,content)        
        col = 14
        content = self.out_recall
        write_excel(file_path,sheet,last_row,col,content)
        col = 15
        content = self.out_precision
        write_excel(file_path,sheet,last_row,col,content)
            
        # WRITE Missing data
        self.mean_truth_table_missing_pts(plot_flag=False)
        mean_TP = np.array(list(self.missing_pts_TP.values())).mean()
        mean_TN = np.array(list(self.missing_pts_TN.values())).mean()
        mean_FP = np.array(list(self.missing_pts_FP.values())).mean()
        mean_FN = np.array(list(self.missing_pts_FN.values())).mean()
        col = 16
        content = mean_TP
        write_excel(file_path,sheet,last_row,col,content)

        col = 17
        content = mean_TN
        write_excel(file_path,sheet,last_row,col,content)        
        col = 18
        content = mean_FP
        write_excel(file_path,sheet,last_row,col,content)        
        col = 19
        content = mean_FN
        write_excel(file_path,sheet,last_row,col,content)
        col = 20
        content = self.miss_accuracy
        write_excel(file_path,sheet,last_row,col,content)        
        col = 21
        content = self.miss_recall
        write_excel(file_path,sheet,last_row,col,content)
        col = 22
        content = self.miss_precision
        write_excel(file_path,sheet,last_row,col,content)
            
        col = 23
        content = comments
        write_excel(file_path,sheet,last_row,col,content)
        col = 24
        content = unique_key
        write_excel(file_path,sheet,last_row,col,content)
        
        col = 25
        content = self.no_match_n
        write_excel(file_path,sheet,last_row,col,content)

    ###########################################################################
    # AUXILIAR
    ###########################################################################

    def correct_format(self,vec_list):
        
        ret_list = list()
        for vec in vec_list:
            ret_list.append(vec.flatten())
        return ret_list
            
    def switch_gt_corr(self,corr_gt):
        template_npts = self.reg_dataset.template.shape[0]
        new_corr = dict()
        for id_ in corr_gt.keys():
            new_corr[id_] = correspondence_switch(corr_gt[id_],template_npts)
        
        return new_corr
    
    def switch_reg_corr(self,corr_vec_dic):
        
        new_corr = dict()
        for id_,corr_vec in corr_vec_dic.items():
            og_shape_sz = self.og_dataset.shapes_dict[id_].points.shape[0]
            new_corr[id_] = correspondence_switch(corr_vec,og_shape_sz)        
        return new_corr

    ###########################################################################
    # DEPRECATED
    ###########################################################################

    def results_corr_vec(self):
        print('Deprecated this is done internally do not call')
        quit()
