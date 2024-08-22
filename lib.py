import matplotlib.pyplot as plt
from skopt.plots import _cat_format,partial_dependence_2D,partial_dependence_1D
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402
from skopt.space import Categorical,Real
from functools import partial
import numpy as np
import skopt 
import sklearn 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pandas import DataFrame
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import List,Dict,Tuple
from skopt.space import Space
from optimizer import ModelOptimizer

def transform_grid(param_grid: Dict
                   ) -> Dict:

    for key, value in param_grid.items():

        if isinstance(param_grid[key],tuple):
            if is_logspaced(np.array(param_grid[key])) :
                mins = min(param_grid[key])
                maxs = max(param_grid[key])
                param_grid[key] = Real(mins,maxs,prior='log-uniform',transform='normalize')
            else:
                mins = min(param_grid[key])
                maxs = max(param_grid[key])
                param_grid[key] = Real(mins,maxs,prior='uniform',transform='normalize')

        # if isinstance(param_grid[key][0],(int,float)) and isinstance(param_grid[key][2],(str)):
        #     continue
        # if isinstance(param_grid[key][0],(int,float)):
        #     param_grid[key] = tuple((min(param_grid[key]),max(param_grid[key])))

        if isinstance(value, list) and not isinstance(param_grid[key][0],(str,int,float,type(None))):
            param_grid[key] = [str(item) for item in value]
        #elif isinstance(value, list) and isinstance(param_grid[key][0],(str,int,float,type(None))):
        # elif isinstance(value, tuple) and not isinstance(param_grid[key][0],(int,float)):
        #     param_grid[key] = [str(item) if not isinstance(item, (str,int,float,type(None))) else item for item in value]
    
    return param_grid



def plot_2D_PDP(feature1 : str,
                feature2 : str,
                features : List,
                samples : np.ndarray,
                plot_dims : List,
                space : skopt.space.space.Space,
                model : sklearn.gaussian_process._gpr.GaussianProcessRegressor
                ):

     fig, ax = plt.subplots()

     index1 = features.index(feature1)
     index2 = features.index(feature2)
     _ ,dim_1 = plot_dims[index1]
     _ ,dim_2 = plot_dims[index2]

     xi, yi, zi = partial_dependence_2D(space, model,
                                                   index1, index2,
                                                   samples, 40)
                                                   
     iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
     if not iscat[index1]:  # bounds not meaningful for categoricals
                    ax.set_ylim(*dim_1.bounds)
     else:
            ax.yaxis.set_major_locator(MaxNLocator(6, integer=iscat[index1]))
            ax.yaxis.set_major_formatter(FuncFormatter(
                        partial(_cat_format, dim_1)))
     if iscat[index2]:
                    ax.xaxis.set_major_locator(MaxNLocator(6,integer=iscat[index2]))
                    # partial() avoids creating closures in a loop
                    ax.xaxis.set_major_formatter(FuncFormatter(
                        partial(_cat_format, dim_2)))
     else:
                    ax.set_xlim(*dim_2.bounds)

        
     im = ax.contourf(xi, yi, zi, 10,
                              cmap='viridis_r')
     ax.set_xlabel(feature2)
     ax.set_ylabel(feature1)
     fig.colorbar(im,label='Accuracy Score')

     
def plot_PDP_1D(features : List,
                space : skopt.space.space.Space,
                samples : np.ndarray ,
                plot_dims : List,
                objectives : List,
                model : List,
                ):
    
    n_dims = len(plot_dims)
    fig, ax = plt.subplots(n_dims, 1,
                           figsize=(2 * n_dims, 2 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.5, wspace=0.1)
    
    if len(objectives) == 1:
        for i in range(n_dims):
            index, dim = plot_dims[i]
            xi, yi = partial_dependence_1D(space, model[0],
                                               index,
                                               samples=samples,
                                               n_points=40)
            yi = [round(yi,2) for yi in yi]
            if n_dims > 1:
                ax_ = ax[i]
            else:
                ax_ = ax
            iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
            if not iscat[i]:
                low, high = dim.bounds
                ax_.set_xlim(low, high)
        #     ax_.plot(xi, yi)
        # else: 
        #      ax_.bar(xi,yi)
            ax_.yaxis.tick_left()
            ax_.yaxis.set_label_position('left')
            ax_.yaxis.set_ticks_position('both')
            ax_.xaxis.tick_bottom()
            ax_.xaxis.set_label_position('bottom')
            ax_.set_xlabel(dim)

            if dim.prior == 'log-uniform':
                ax_.set_xscale('log')
            else:
                ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
            if iscat[i]:
                ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim)))
            ax_.plot(xi, yi)

            fig.suptitle('Partial Dependence Plots for each Hyperparameter')
            ax_.set_xlabel(features[i])
            ax_.set_ylabel(objectives[0])
    elif len(objectives) == 2:
        for i in range(n_dims):
            index, dim = plot_dims[i]
            xi1, yi1 = partial_dependence_1D(space, model[0],
                                               index,
                                               samples=samples,
                                               n_points=40)
            yi1 = [round(yi1,2) for yi1 in yi1]

            xi2, yi2 = partial_dependence_1D(space, model[1],
                                               index,
                                               samples=samples,
                                               n_points=40)
            yi2 = [round(yi2,4) for yi2 in yi2]
            if n_dims > 1:
                ax_ = ax[i]
            else:
                ax_ = ax
            iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
            if not iscat[i]:
                low, high = dim.bounds
                ax_.set_xlim(low, high)
        #     ax_.plot(xi, yi)
        # else: 
        #      ax_.bar(xi,yi)
            ax_.yaxis.tick_left()
            ax_.yaxis.set_label_position('left')
            ax_.yaxis.set_ticks_position('both')
            ax_.xaxis.tick_bottom()
            ax_.xaxis.set_label_position('bottom')
            ax_.set_xlabel(dim)

            if dim.prior == 'log-uniform':
                ax_.set_xscale('log')
            else:
                ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
            if iscat[i]:
                ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim)))
            ax_.plot(xi1, yi1,label=objectives[0])
            ax2 = ax_.twinx()
            ax2.plot(xi2,yi2,'r',label=objectives[1],alpha=0.5)
            fig.suptitle('Partial Dependence Plots for each Hyperparameter')
            ax_.set_xlabel(features[i])
            ax_.set_ylabel(objectives[0])       
            ax2.set_ylabel(objectives[1])
            lines, labels = ax_.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()  
            ax2.legend(lines + lines2, labels + labels2, loc='lower right')     
    else:
        for i in range(n_dims):
            index, dim = plot_dims[i]
            xi1, yi1 = partial_dependence_1D(space, model[0],
                                               index,
                                               samples=samples,
                                               n_points=40)
            yi1 = [round(yi1,2) for yi1 in yi1]

            xi2, yi2 = partial_dependence_1D(space, model[1],
                                               index,
                                               samples=samples,
                                               n_points=40)
            yi2 = [round(yi2,4) for yi2 in yi2]

            xi3, yi3 = partial_dependence_1D(space, model[2],
                                               index,
                                               samples=samples,
                                               n_points=40)
            yi3 = [round(yi3,4) for yi3 in yi3]
            if n_dims > 1:
                ax_ = ax[i]
            else:
                ax_ = ax
            iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
            if not iscat[i]:
                low, high = dim.bounds
                ax_.set_xlim(low, high)
        #     ax_.plot(xi, yi)
        # else: 
        #      ax_.bar(xi,yi)
            ax_.yaxis.tick_left()
            ax_.yaxis.set_label_position('left')
            ax_.yaxis.set_ticks_position('both')
            ax_.xaxis.tick_bottom()
            ax_.xaxis.set_label_position('bottom')
            ax_.set_xlabel(dim)

            if dim.prior == 'log-uniform':
                ax_.set_xscale('log')
            else:
                ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
            if iscat[i]:
                ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim)))
                
            ax_.plot(xi1, yi1,label=objectives[0])
            ax2 = ax_.twinx()
            ax2.plot(xi2,yi2,'r',label=objectives[1],alpha=0.5)
            ax3 = ax_.twinx()
            ax3.plot(xi3,yi3,'g',label=objectives[2],alpha=0.5)
            ax3.spines.right.set_position(("axes", 1.2))

            fig.suptitle('Partial Dependence Plots for each Hyperparameter')

            ax_.set_xlabel(features[i])
            ax_.set_ylabel(objectives[0])       
            ax2.set_ylabel(objectives[1])
            ax3.set_ylabel(objectives[2])
            

            lines, labels = ax_.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()  
            lines3, labels3 = ax3.get_legend_handles_labels()  
            ax2.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='lower right')

        

def compute_scaled_datasets(data : DataFrame,
                            scaler : List = None,
                            imputer : List = None
                            ) -> Dict:

    numeric_cols=data.select_dtypes(include=np.number).columns
    scalers = {str(scaler): scaler for scaler in scaler}
    if imputer != None:
        imputers = {str(imputer): imputer for imputer in imputer}

    # Create an empty dictionary to store the scaled datasets
    scaled_datasets = {}

    # Iterate over the imputers
    if imputer != None:
        
        for imputer_name, imputer in imputers.items():
        # Iterate over the scalers
            for scaler_name, scaler in scalers.items():
            # Create a pipeline with the imputer and scaler
                num_transform = Pipeline([
                    ('imputer', SimpleImputer(strategy=imputer)),
                    ('scaler', scaler)
                ])
            
                preprocessor = ColumnTransformer(
                    transformers=[
                    ("num_transform", num_transform, numeric_cols)])
            # Fit and transform the pipeline on the data
                scaled_X = preprocessor.fit_transform(data)

            # Create a DataFrame with the scaled data
                scaled_df = pd.DataFrame(scaled_X.round(3), columns=data.columns)

            # Store the scaled dataset in the dictionary with a unique key
                key = f'{imputer_name}_{scaler_name}'
                scaled_datasets[key] = scaled_df
    else:
        for scaler_name, scaler in scalers.items():
            # Create a pipeline with the imputer and scaler
                num_transform = Pipeline(steps = [
                    ('scaler', scaler)
                ])
                preprocessor = ColumnTransformer(
                    transformers=[
                    ("num_transform", num_transform, numeric_cols)])
                
            # Fit and transform the pipeline on the data
                scaled_X = preprocessor.fit_transform(data)

            # Create a DataFrame with the scaled data
                scaled_df = pd.DataFrame(scaled_X.round(3), columns=data.columns)

            # Store the scaled dataset in the dictionary with a unique key
                key = f'{scaler_name}'
                scaled_datasets[key] = scaled_df


    return scaled_datasets

def dimensions_aslists(search_space : Dict
                       ):
    """Convert a dict representation of a search space into a list of
    dimensions, ordered by sorted(search_space.keys()).

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        :class:`skopt.space.Dimension` (Real, Integer or Categorical)

    Returns
    -------
    params_space_list: list
        list of skopt.space.Dimension instances
    """
    params_space_list = [
        search_space[k] for k in sorted(search_space.keys())
    ]
    name = [
        k for k in sorted(search_space.keys())
    ]
    return params_space_list,name

def transform_samples(hyperparameters : List[Dict],
                      space : Space,
                      name: List
                      ) -> np.ndarray:
    rearranged_list = []

    for dictionary in hyperparameters:
        rearranged_dict = {key: dictionary[key] for key in name}
        rearranged_list.append(rearranged_dict)


    spaces = [list(rearranged_list[i].values()) for i in range(len(hyperparameters))]
    for sublist in spaces:
        for i in range(len(sublist)):
            if not isinstance(sublist[i], (int,float,str,type(None))):
                sublist[i] = str(sublist[i])
    samples = space.transform(spaces)

    return samples

def gaussian_objective(objective : str, 
                       optimizer : ModelOptimizer,
                       samples : np.ndarray):

    if objective == 'accuracy':
        gaussian = pd.DataFrame(samples)
        gaussian['label'] = optimizer.cv_results_['mean_test_score']
        gaussian = gaussian.dropna().reset_index(drop=True)
    elif objective == 'fit_time':
        gaussian = pd.DataFrame(samples)
        gaussian['label'] = optimizer.cv_results_['mean_fit_time']
        gaussian = gaussian[gaussian.label !=0 ]
    elif objective == 'score_time':
        gaussian = pd.DataFrame(samples)
        gaussian['label'] = optimizer.cv_results_['mean_score_time']
        gaussian = gaussian[gaussian.label !=0 ]

    X = gaussian.drop(columns='label')
    y = gaussian['label']

    return X,y

def is_logspaced(arr):
    ratios = arr[1:] / arr[:-1]
    return np.allclose(ratios, ratios[0])

         
