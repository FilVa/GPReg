# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:55:57 2019

@author: filipavaldeira
"""


import matplotlib.pyplot as plt
import numpy as np

##########################
### Vertical plot bar ####
##########################
def vertical_bar_plot(data, ax, label, title, xlabel, ylabel):
    
    index = np.arange(len(label))
    ax.bar(index, data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set(title = title)
    ax.set_xticks(index, label)
    

##########################
######## Line plot #######
########################## 
def line_plot(data, ax, label, title, xlabel, ylabel):
    
    index = np.arange(len(label))
    ax.plot(index, data)
    ax.set_xlabel(xlabel, fontsize=5)
    ax.set_ylabel(ylabel, fontsize=5)
    ax.set_xticks(index, label)
    ax.set(title = title)




##########################
######## Table plot #######
########################## 
    
def table_plot(ax,data,cols,rows,title):
    
    
    
    ax.table(cellText=data,
                      rowLabels=rows,
                      colLabels=cols,
                      loc='center')
    ax.axis("off")
    ax.set(title = title)

##########################
######## Truth table plot #######
##########################
def truth_table_plot(data_tab,data_metrics,title_,cols_name,rows_name):

    TP,TN,FP,FN = data_tab
    accuracy, recall, precision = data_metrics

    cols = cols_name
    rows = rows_name
    data = [[ TP, FP],[ FN,TN]]
    title = title_
    
    fig, ax = plt.subplots()
    table_plot(ax,data,cols,rows,title)
    
    s = 'Accuracy =  {} '.format(accuracy)
    ax.text(0.1,0.3,s)
    s = 'Recall =  {} '.format(recall)
    ax.text(0.1,0.25,s)        
    s = 'Precision =  {} '.format(precision)
    ax.text(0.1,0.2,s)    
 
    plt.show()
    
    return accuracy,recall,precision


def plot_dfResults(df_results_raw):
    
    df_results = df_results_raw.fillna(-.1).replace(0,1e-2)

    fig, axs = plt.subplots(5, 2, figsize=(10,20))
    
    # set top and right frame not visible
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    
    df_results['Success'].plot.bar(ax=axs[0,0],rot=0)
    #axs[0].get_legend().remove()
    axs[0,0].set_ylim([0,1])
    axs[0,0].set_title('Fraction of successful registration')
    
    df_results[['Miss_acc','Out_acc']].plot.bar(ax=axs[0,1],rot=0)
    axs[0,1].legend(frameon=False)
    axs[0,1].set_ylim([0,1])
    axs[0,1].set_title('Accurary for missing data and outliers')
    
    
    df_results['Frac_corr'].plot.bar(ax=axs[1,0],rot=0)
    axs[1,0].set_ylim([0,1])
    axs[1,0].set_title('Total correspondences found as fraction of GT')
    df_results['Exact_corr'].plot.bar(ax=axs[1,1],rot=0)
    axs[1,1].set_ylim([0,1])
    axs[1,1].set_title('Exact correspondences found as fraction of GT')
    
    
    
    df_results[['Miss_recall','Miss_precision']].plot.bar(ax=axs[3,0],rot=0)
    axs[3,0].legend(frameon=False)
    axs[3,0].set_ylim([-0.2,1])
    axs[3,0].set_title('Missing data metrics over dataset')
    df_results[['Out_recall','Out_precision']].plot.bar(ax=axs[3,1],rot=0)
    axs[3,1].legend(frameon=False)
    axs[3,1].set_ylim([-0.2,1])
    axs[3,1].set_title('Outlier data metrics over dataset')
    
    
    df_results['Err_target'].plot.bar(ax=axs[2,0],rot=0)
    axs[2,0].set_title('Distance error in original target')
    df_results['Err_def_template'].plot.bar(ax=axs[2,1],rot=0)
    axs[2,1].set_title('Distance error in deformed template')
    
    # plot horizontal line with original template error
    x = np.linspace(-0.5,df_results.shape[1],100)
    y = np.ones(x.shape)*df_results['Err_og_template'][0]
    axs[2,1].plot(x,y,'--k',label = 'Original template')
    axs[2,1].legend()
    axs[2,1].legend(frameon=False)
    axs[2,1].set_yscale('log')
    
    df_results['Time_per_shape'].plot.bar(ax=axs[4,0],rot=0)
    axs[4,0].set_title('Time per shape in seconds')  
    

    fig.delaxes(axs[4,1])
    # Add notes to plot
    text = 'NOTE :\n   - nan values are represented as -0.1\n   - zero values are represented with a small epsilon'
    axs[3,1].annotate(text, xy=(-0.5, -1), xycoords='data', annotation_clip=False)
    
    return fig, axs


def plot_LevelMethod(list_results,list_labels):

    fig, axs = plt.subplots(5, 2, figsize=(10,30))
    x_labels = list_results[0].index
    marker_list = ['o','x','s','.','*','+','o','*','*','*','+']
        
    for df_results_raw,df_label,marker in zip(list_results,list_labels,marker_list):
        df_results = df_results_raw.fillna(-.1).replace(0,1e-2)
        
        df_results['Success'].plot(ax=axs[0,0],rot=0,label=df_label,marker=marker)
        #axs[0].get_legend().remove()
        axs[0,0].set_ylim([0,1])
        axs[0,0].set_title('Fraction of successful registration')
        
        df_results['Time_per_shape'].plot(ax=axs[0,1],rot=0,label=df_label,marker=marker)
        axs[0,1].set_title('Time per shape in seconds')  
        
        # df_results[['Miss_acc','Out_acc']].plot(ax=axs[0,1],rot=0)
        # axs[0,1].legend(frameon=False)
        # axs[0,1].set_ylim([0,1])
        # axs[0,1].set_title('Accurary for missing data and outliers')
        
        
        df_results['Frac_corr'].plot(ax=axs[1,0],rot=0,label=df_label,marker=marker)
        axs[1,0].set_ylim([0,1])
        axs[1,0].set_title('Total correspondences found as fraction of GT')
        df_results['Exact_corr'].plot(ax=axs[1,1],rot=0,label=df_label,marker=marker)
        axs[1,1].set_ylim([0,1])
        axs[1,1].set_title('Exact correspondences found as fraction of GT')
        
        
        

        
        
        df_results['Err_target'].plot(ax=axs[2,0],rot=0,label=df_label,marker=marker)
        axs[2,0].set_title('Distance error in original target')
        df_results['Err_def_template'].plot(ax=axs[2,1],rot=0,label=df_label,marker=marker)
        axs[2,1].set_title('Distance error in deformed template')
        
        
        df_results['Miss_recall'].plot(ax=axs[3,0],rot=0,label=df_label,marker=marker)
        axs[3,0].set_ylim([-0.2,1])
        axs[3,0].set_title('Missing recall over dataset')
        df_results['Miss_precision'].plot(ax=axs[3,1],rot=0,label=df_label,marker=marker)
        axs[3,1].set_ylim([-0.2,1])
        axs[3,1].set_title('Missing precision over dataset')
        
        df_results['Out_recall'].plot(ax=axs[4,0],rot=0,label=df_label,marker=marker)
        axs[4,0].set_ylim([-0.2,1])
        axs[4,0].set_title('Outliers recall over dataset')
        df_results['Out_precision'].plot(ax=axs[4,1],rot=0,label=df_label,marker=marker)
        axs[4,1].set_ylim([-0.2,1])
        axs[4,1].set_title('Outliers precision over dataset')
        
    
    # Common settings for axis
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x_labels)
        ax.legend(frameon=False)
        ax.set_xlabel('Missing data level')
        
    #fig.delaxes(axs[4,1])
    # Add notes to plot
    text = 'NOTE :\n   - nan values are represented as -0.1\n   - zero values are represented with a small epsilon'
    axs[4,0].annotate(text, xy=(1, -0.8), xycoords='data', annotation_clip=False)
    
    return fig, axs



def plot_PAPER_MISSING(list_results,list_labels, xlabel):

    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    x_labels = list_results[0].index
    marker_list = ['o','x','s','.','*','+','o','*','+','*','+']
    linestyle_list = ['solid','solid','solid','dashed','dotted','dotted','dotted']
    color_list= ['red', 'orange', 'blue','green','brown','black','grey']
    
    for df_results_raw,df_label,marker, linestyle, col in zip(list_results,list_labels,marker_list,linestyle_list,color_list):
        df_results = df_results_raw.fillna(-.1).replace(0,1e-2)
        
        if df_label == 'GPReg_Full':
            df_results['Miss_recall'].plot(ax=axs[0],rot=0,label=df_label,color='red',linewidth = 5)
            #axs[0].get_legend().remove()
            axs[0].set_ylabel('Missing recall')
    
            df_results['Miss_precision'].plot(ax=axs[1],rot=0,color='red',linewidth = 5)
            axs[1].set_ylabel('Missing precision')
           
        else:
            df_results['Miss_recall'].plot(ax=axs[0],rot=0,label=df_label,marker=marker,linestyle=linestyle,color = col)
            #axs[0].get_legend().remove()
            axs[0].set_ylabel('Missing recall')
    
            df_results['Miss_precision'].plot(ax=axs[1],rot=0,marker=marker,linestyle=linestyle,color = col)
            axs[1].set_ylabel('Missing precision')
        
    
    # Common settings for axis
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x_labels)
        ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylim([0,1])
    
    axs[1].legend().remove()
    
    return fig, axs

def plot_PAPER(list_results,list_labels, xlabel):

    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    x_labels = list_results[0].index
    marker_list = ['o','x','s','.','*','+','o']
    linestyle_list = ['solid','solid','solid','dashed','dotted','dotted','dotted']
    color_list= ['red', 'orange', 'blue','green','brown','black','grey']
    
    for df_results_raw,df_label,marker, linestyle, col in zip(list_results,list_labels,marker_list,linestyle_list,color_list):
        df_results = df_results_raw.fillna(-.1).replace(0,1e-2)
        
        if df_label == 'GPReg_Full':
            df_results['Success'].plot(ax=axs[0],rot=0,label=df_label,color='red',linewidth = 5)
            #axs[0].get_legend().remove()
            axs[0].set_ylim([0,1])
            axs[0].set_ylabel('Fraction of successful registration')
    
            df_results['Err_def_template'].plot(ax=axs[1],rot=0,color='red',linewidth = 5)
            axs[1].set_ylabel('Distance error')
           
        else:
            df_results['Success'].plot(ax=axs[0],rot=0,label=df_label,marker=marker,linestyle=linestyle,color = col)
            #axs[0].get_legend().remove()
            axs[0].set_ylim([0,1])
            axs[0].set_ylabel('Fraction of successful registration')
    
            df_results['Err_def_template'].plot(ax=axs[1],rot=0,marker=marker,linestyle=linestyle,color = col)
            axs[1].set_ylabel('Distance error')
        
    
    # Common settings for axis
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x_labels)
        ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
    
    axs[1].legend().remove()
    
    return fig, axs

def plot_OneLevel(list_results,list_labels):

    fig, axs = plt.subplots(5, 2, figsize=(10,30))
    x_labels = list_results[0].index
    ids = np.arange(len(list_labels))
    
    for df_results_raw,df_label,i in zip(list_results,list_labels,ids):
        df_results = df_results_raw.fillna(-.1).replace(0,1e-2)
        
        axs[0,0].bar(i,df_results['Success'])
        
        #axs[0].get_legend().remove()
        axs[0,0].set_ylim([0,1])
        axs[0,0].set_title('Fraction of successful registration')
        
        axs[0,1].bar(i,df_results['Time_per_shape'],label= df_label)
        axs[0,1].set_title('Time per shape in seconds')  
        

        axs[1,0].bar(i,df_results['Frac_corr'],label= df_label)
        axs[1,0].set_ylim([0,1])
        axs[1,0].set_title('Total correspondences found as fraction of GT')
        axs[1,1].bar(i,df_results['Exact_corr'],label= df_label)
        axs[1,1].set_ylim([0,1])
        axs[1,1].set_title('Exact correspondences found as fraction of GT')
        
        
        axs[2,0].bar(i,df_results['Err_target'],label= df_label)
        axs[2,0].set_title('Distance error in original target')
        axs[2,1].bar(i,df_results['Err_def_template'])
        axs[2,1].set_title('Distance error in deformed template')
        
        axs[3,0].bar(i,df_results['Miss_recall'],label= df_label)
        axs[3,0].set_ylim([-0.2,1])
        axs[3,0].set_title('Missing recall over dataset')
        axs[3,1].bar(i,df_results['Miss_precision'],label= df_label)
        axs[3,1].set_ylim([-0.2,1])
        axs[3,1].set_title('Missing precision over dataset')
        
        axs[4,0].bar(i,df_results['Out_recall'],label= df_label)
        axs[4,0].set_ylim([-0.2,1])
        axs[4,0].set_title('Outliers recall over dataset')
        axs[4,1].bar(i,df_results['Out_precision'],label= df_label)
        axs[4,1].set_ylim([-0.2,1])
        axs[4,1].set_title('Outliers precision over dataset')
        
    
    
    # Common settings for axis
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x_labels)
        ax.legend(frameon=False)
        ax.set_xlabel('Missing data level')
        
    #fig.delaxes(axs[4,1])
    # Add notes to plot
    text = 'NOTE :\n   - nan values are represented as -0.1\n   - zero values are represented with a small epsilon'
    axs[4,0].annotate(text, xy=(1, -0.8), xycoords='data', annotation_clip=False)
    
    return fig, axs