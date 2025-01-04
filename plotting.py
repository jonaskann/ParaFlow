''' Function with all the relevant plotting functions. '''

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import mplhep as hep
plt.rcdefaults()
plt.style.use([hep.style.ROOT])

from utils import get_binedges

from scipy.stats import ks_2samp, binned_statistic_2d


def chi2(hist1, hist2):
        
        ''' function for calculating the chi2 value'''

        n_bins = len(hist1)

        error = np.sqrt(hist1 + hist2) # poisson error on both histograms
        epsilon = 1e-8 # to avoid division by zero

        return np.sum((((hist1 - hist2) ** 2) / (error + epsilon)**2)) / n_bins


def plot_histogram(data_samples, data_geant4, bin_centers, plot_color, error_color, title, x_label, y_label, filename, result_path, data_conditions, sample_conditions, xlim = None, ylim = None, calc_std = False, thickness_range = None, distance_range = None, bin_comparison_thickness = False, bin_comparison_distance = False, log_yscale = False, hist_overflow = True):

    ''' Function for plotting the histogram of various quantities including the comparison histograms.'''


    bin_edges = get_binedges(bin_centers=bin_centers)

    hist_samples, _ = np.histogram(data_samples, bins = bin_edges)
    hist_data, _            = np.histogram(data_geant4, bins = bin_edges)

    # Add histogram overflow if specified
    if ( hist_overflow ):

        overflow_samples = np.sum(data_samples > bin_edges[-1])
        overflow_data    = np.sum(data_geant4  > bin_edges[-1])

        underflow_samples = np.sum(data_samples < bin_edges[0])
        underflow_data    = np.sum(data_geant4  < bin_edges[0])

        hist_samples[0] += underflow_samples
        hist_data[0] += underflow_data

        hist_samples[-1] += overflow_samples
        hist_data[-1] += overflow_data


    ### Statistical scores ###
    chi2_value = chi2(hist1=hist_data, hist2=hist_samples)

    # Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(data_samples, data_geant4, alternative = 'two-sided')


    
    ### Plotting ...

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex = True, gridspec_kw={'hspace': 0.1, 'height_ratios': [7, 1.5, 1.5]})

    
    ax1.set_title("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$" , c="black", fontsize = 35, loc = 'left')

    # Add an additional title if a specific bin is evaluated
    if (thickness_range == (0.5,1.5)) and (distance_range == (50,90)):
        condition_title = "" 
    
    elif (thickness_range != (0.5,1.5)):
        condition_title = f"$\\bf{{d \\in ({thickness_range[0]},{thickness_range[1]})~X_0}}$" 
    else:
        condition_title = f"$\\bf{{b \\in ({distance_range[0]},{distance_range[1]})~cm}}$"

    ax1.set_title(condition_title, c = "dimgrey", fontsize =25, loc = 'right')

    epsilon = 1e-4

    # Histogram bars of Geant4 data
    ax1.fill_between(bin_edges, np.append(hist_data, 0), step='post', color=plot_color, alpha = 0.7, label = r"MC", edgecolor = 'black')
    #ax1.bar(bin_centers, hist_data, align = 'center', width =np.diff(bin_edges)+epsilon, color = plot_color, alpha = 0.7, label = r"MC", edgecolor = None)


    # Uncertainties as grid patches
    uncertainties_data = np.sqrt(hist_data)
    ax1.bar(
        bin_centers,
        uncertainties_data,
        bottom=hist_data - uncertainties_data/2,
        width=np.diff(bin_edges),
        align='center',
        color='none',
        alpha=0.8,
        edgecolor=error_color,
        hatch = 'xxxxxxx',
        label='Stat. unc.',
        linewidth = 0.0,
    )

    # Samples as data points with error bars
    ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt='.', markersize = 12, markeredgecolor ='black', markerfacecolor = 'black', ecolor='black', elinewidth = 2, label = "ParaFlow")


    # Set axis limits as specified or based on data
    if xlim: ax1.set_xlim(*xlim)
    ax1.set_ylim(0, max(max(hist_samples), max(hist_data))*1.5)

    # if log_yscale is active 
    if log_yscale: 
        ax1.set_yscale('log')
    
    if ylim: ax1.set_ylim(*ylim)


    ax1.set_ylabel(y_label, fontsize = 35)
    ax1.legend(loc = 'upper left', fontsize = 30)
    ax1.tick_params(axis='both', which='major', labelsize=30)


    # Add the relevant statistical scores to the plot as text box
    text = f'$\chi^2 / n_{{\mathrm{{bins}}}} = {chi2_value:.2f}$\n$KS = {ks_stat:.4f}$\n$p_{{KS}} = {ks_p_value:.4f}$'

    props = dict(boxstyle="square, pad=0.3", edgecolor="black", facecolor="white")
    ax1.text(
        0.65, 0.96,  # Coordinates relative to the axes
        text, 
        transform=ax1.transAxes, 
        fontsize=30, 
        bbox=props,
        verticalalignment='top',  # Align box to the top
        horizontalalignment='left',  # Align box to the right
    )

    ####### RATIO PLOTS #######

    # Calculate the ratio, avoiding division by zero
    ratio = np.divide(hist_samples, hist_data, out=np.full_like(hist_data, fill_value= 0, dtype=float), where=hist_data != 0)

    inv_hist_data = np.divide(np.ones_like(hist_data, dtype=float), hist_data, out=np.zeros_like(hist_data, dtype=float), where=hist_data != 0)
    inv_hist_samples = np.divide(np.ones_like(hist_samples, dtype=float), hist_samples, out=np.zeros_like(hist_samples, dtype=float), where=hist_samples != 0)

    err_data = hist_samples * (inv_hist_data)**2 * uncertainties_data

    err_samples = np.sqrt(hist_samples) * inv_hist_data

    # set values with ratio 0 (which means hist_samples = 0) to -1 out of sight
    ratio[ratio == 0] = -1



    # Plot the ratio on the 1. lower subplot
    ax2.axhline(1, color='black', linestyle='--')  # Reference line at y=1
    ax2.errorbar(bin_centers, ratio, yerr = err_samples, fmt = '.', color='black', elinewidth= 2.5, markersize = 12, ecolor = 'black')
    ax2.bar(
        bin_centers,
        err_data,
        bottom= 1 - err_data/2,
        width=np.diff(bin_edges),
        align='center',
        color='none',
        alpha=0.8,
        edgecolor=error_color,
        hatch = 'xxxxxxx',
        linewidth = 0.0,
    )
    

    ax2.set_ylabel('')
    ymin, ymax = 0, 2
    ax2.set_ylim(ymin, ymax)  # Adjust y-limits for better view

    # Add arrows for values outside y-axis range
    for i, (xi, yi) in enumerate(zip(bin_centers, ratio)):
        if yi > ymax:
            ax2.annotate('', xy=(xi, ymax - 0.1), xytext=(xi, ymax - 0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

    # ylabel
    fig.text(0.02, 0.23, r'ParaFlow / MC', va='center', rotation='vertical', fontsize = 35)


    # Plot the finer ratio on the lowest subplot

    # lines at intervals of 5 % deviation
    ax3.axhline(1, color='black', linestyle='--')  # Reference line at y=1
    ax3.axhline(1.05, color='grey')  # Reference line at y=1.01
    ax3.axhline(0.95, color='grey')
    ax3.axhline(1.10, color='grey', ls = '--')
    ax3.axhline(0.90, color='grey', ls = '--')

    ax3.errorbar(bin_centers, ratio, yerr = err_samples, fmt = '.', color='black', elinewidth= 2.5, markersize = 12, ecolor = 'black')
    ax3.bar(
        bin_centers,
        err_data,
        bottom= 1 - err_data/2,
        width=np.diff(bin_edges),
        align='center',
        color='none',
        alpha=0.8,
        edgecolor=error_color,
        hatch = 'xxxxxxx',
        linewidth = 0.0,
    )

    ax3.set_xlabel(x_label, fontsize = 35)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax3.tick_params(axis='both', which='major', labelsize=30)


    ax3.set_ylabel('')
    ymin, ymax = 0.80, 1.20
    ax3.set_ylim(ymin, ymax)  # Adjust y-limits for better view
    
    fig.savefig(result_path + filename + ".pdf", bbox_inches='tight')


    #########################################################################################################
    #        Add another plot, where we compare the histograms for the different thickness bins             #
    #########################################################################################################

    if (bin_comparison_thickness):

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex = True, gridspec_kw={'hspace': 0.1, 'height_ratios': [7, 1.5, 1.5]})


        thickness_bins = [(0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5)]
        plot_color = ['orange', 'red', 'royalblue', 'darkmagenta']

        # for legends
        custom_handles_legend1 = []
        custom_handles_legend2 = []

        # running max for y-limit
        running_max = 0

        for i, thickness_range in enumerate(thickness_bins):

            shield_label = f"$d \\in ({thickness_range[0]},{thickness_range[1]})~X_0$" 


            mask_samples = (sample_conditions[:,1] < thickness_range[1]) & (sample_conditions[:,1] > thickness_range[0])
            mask_data    = (data_conditions[:,1] < thickness_range[1]) & (data_conditions[:,1] > thickness_range[0])

            data_samples_filtered = data_samples[mask_samples]
            data_geant4_filtered  = data_geant4[mask_data]

            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(data_samples_filtered, data_geant4_filtered, alternative = 'two-sided')

            hist_samples, _ = np.histogram(data_samples_filtered, bins = bin_edges)
            hist_data, _            = np.histogram(data_geant4_filtered, bins = bin_edges)



            ax1.set_title("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$", c="black", fontsize = 35, loc = 'left', pad = 15)
            ax1.step(np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[1] - bin_edges[0])), np.insert(np.append(hist_data, 0),0,0), where='post', color=plot_color[i], lw = 3.5, alpha = 0.6)
            ax1.fill_between(bin_edges, np.append(hist_data, 0), step='post', color=plot_color[i], edgecolor = plot_color[i], lw = 0, alpha = 0.025)
            ax1.errorbar(bin_centers, hist_data, yerr = np.sqrt(hist_data), fmt='none', ecolor=plot_color[i], alpha = 0.4, elinewidth = 2)
            ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt = ".", color = plot_color[i], ecolor = plot_color[i], markersize = 16, elinewidth=2.5, markeredgecolor = 'black', markeredgewidth = 0.25)
            

            # Add thickness info to custom legend
            custom_handles_legend1.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = shield_label))
            custom_handles_legend2.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = f"$KS = {ks_stat:.3f},~p_{{KS}} = {ks_p_value:.3f}$"))

            ax1.set_ylabel("Events", fontsize = 35)


            ### Ratio Plot....


            # Calculate the ratio, avoiding division by zero
            ratio = np.divide(hist_samples, hist_data, out=np.full_like(hist_data, fill_value= 0, dtype=float), where=hist_data != 0)

            inv_hist_data = np.divide(np.ones_like(hist_data, dtype=float), hist_data, out=np.zeros_like(hist_data, dtype=float), where=hist_data != 0)
            inv_hist_samples = np.divide(np.ones_like(hist_samples, dtype=float), hist_samples, out=np.zeros_like(hist_samples, dtype=float), where=hist_samples != 0)

            # here perform error propagation for single error bars
            err = ratio * np.sqrt(inv_hist_data+ inv_hist_samples)

            ratio[ratio == 0] = -1

            bin_width = bin_centers[1] - bin_centers[0]


            # Determine offsets between the data points to improve visibility
            n_ratios = len(thickness_bins)
            if n_ratios == 1:
                offsets = [0.0]
            elif n_ratios == 2:
                offsets = [-0.1 * bin_width, 0.1 * bin_width]
            else:
                offsets = np.linspace(-0.1 * bin_width, 0.1 * bin_width, n_ratios)



            # Plot the ratio on the 1. lower subplot
            ax2.errorbar(bin_centers + offsets[i], ratio, yerr = err, fmt = '.', color=plot_color[i], elinewidth= 2.5, markersize = 12, ecolor = plot_color[i], markeredgecolor = 'black', markeredgewidth = 0.25)
            ax2.axhline(1, color='black', linestyle='--')  # Reference line at y=1

            ax2.set_ylabel('')
            ymin, ymax = 0, 2
            ax2.set_ylim(ymin, ymax) 

            # Add arrows for values outside y-axis range
            for _, (xi, yi) in enumerate(zip(bin_centers, ratio)):
                if yi > ymax:
                    ax2.annotate('', xy=(xi, ymax - 0.1), xytext=(xi, ymax - 0.5),
                                arrowprops=dict(arrowstyle='->', color=plot_color[i], lw=2.5))

            # ylabel
            fig.text(0.02, 0.23, r'ParaFlow / MC', va='center', rotation='vertical', fontsize = 35)
                    
            # Plot the finer ratio on the lowest subplot
            ax3.errorbar(bin_centers + offsets[i], ratio, yerr = err, fmt = '.', color=plot_color[i], elinewidth= 2.5, markersize = 12, ecolor = plot_color[i], markeredgecolor = 'black', markeredgewidth = 0.25)
            ax3.axhline(1, color='black', linestyle='--')  # Reference line at y=1
            ax3.axhline(1.05, color='grey')  # Reference line at y=1.01
            ax3.axhline(0.95, color='grey')
            ax3.axhline(1.10, color='grey', ls = '--')
            ax3.axhline(0.90, color='grey', ls = '--')

                
            # running max for getting the right y-axis limits
            running_max = max((max(hist_samples), max(hist_data), running_max))


        # first legend with info regarding thickness
        first_legend = ax1.legend(handles = custom_handles_legend1, loc = "upper right", fontsize = 27)
        ax1.add_artist(first_legend)


        # legend with statistical scores
        legend_properties = {'weight':'bold', 'size':27}
        second_legend = ax1.legend(handles = custom_handles_legend2, handlelength=0, handletextpad=0, loc = 'lower left', labelcolor = 'linecolor', fontsize = 27, prop = legend_properties, facecolor = "white", framealpha = 1, edgecolor = 'black', bbox_to_anchor=(0.18, 0.0075), frameon = True, fancybox = False)
        
        # Customize the frame around the legend
        frame = second_legend.get_frame()
        frame.set_edgecolor('black')  # Set the edge color to black
        frame.set_linewidth(2) 
        frame.set_facecolor('white')  # Ensure the background is not transparent

        for item in second_legend.legend_handles:
            item.set_visible(False)
        ax1.add_artist(second_legend)


        # second legend to distinguish MC from FastSim
        ax1.errorbar(np.inf, np.inf, yerr = [1], color = "dimgrey", label = "ParaFlow", fmt=".", markersize = 15) # dump plot for legend
        ax1.errorbar(np.inf, np.inf, yerr = [1], color ="dimgrey", markersize = 0, elinewidth = 1, label = "MC", lw = 3.5) # dump plot for legend
        ax1.legend(loc = "upper left", fontsize = 30)
        
        

        # set axis limits
        ax1.set_ylim(0, running_max*1.5)
        if xlim: ax1.set_xlim(*xlim)
        if ylim: ax1.set_ylim(*ylim)

        # log_yscale if needed
        if log_yscale: 
            ax1.set_yscale('log')


        # axis labels
        ax1.set_ylabel(y_label, fontsize = 35)
        ax3.set_ylabel('')
        ymin, ymax = 0.80, 1.20
        ax3.set_ylim(ymin, ymax)
        ax3.set_xlabel(x_label, fontsize = 35)

        ax1.tick_params(axis='both', which='major', labelsize=30)
        ax2.tick_params(axis='both', which='major', labelsize=30)
        ax3.tick_params(axis='both', which='major', labelsize=30)
        


        fig.savefig(result_path + "bin_comparison_thickness/comparison_" + filename + ".pdf", bbox_inches = 'tight')

        plt.close('all')


    # Now repeat for comparison plots of distance ranges
    if (bin_comparison_distance):

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex = True, gridspec_kw={'hspace': 0.1, 'height_ratios': [7, 1.5, 1.5]})


        distance_bins = [(50, 60), (60, 70), (70, 80), (80, 90)]
        plot_color = ['orange', 'red', 'royalblue', 'darkmagenta']
        custom_handles_legend1 = []
        custom_handles_legend2 = []

        # running max for y limit
        running_max = 0

        for i, distance_range in enumerate(distance_bins):

            distance_label = f"$b \\in ({distance_range[0]},{distance_range[1]})~cm$" 

            mask_samples = (sample_conditions[:,2] < distance_range[1]) & (sample_conditions[:,2] > distance_range[0])
            mask_data    = (data_conditions[:,2] < distance_range[1]) & (data_conditions[:,2] > distance_range[0])

            data_samples_filtered = data_samples[mask_samples]
            data_geant4_filtered  = data_geant4[mask_data]

            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(data_samples_filtered, data_geant4_filtered, alternative = 'two-sided')

            hist_samples, _ = np.histogram(data_samples_filtered, bins = bin_edges)
            hist_data, _            = np.histogram(data_geant4_filtered, bins = bin_edges)

            
            ### Subplot 1

            ax1.set_title("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$", c="black", fontsize = 35, loc = 'left', pad = 15)

            
            ax1.fill_between(bin_edges, np.append(hist_data, 0), step='post', color=plot_color[i], edgecolor = plot_color[i], lw = 0, alpha = 0.025)
            ax1.step(np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[1] - bin_edges[0])), np.insert(np.append(hist_data, 0),0,0), where='post', color=plot_color[i], lw = 3.5, alpha = 0.6)
            ax1.errorbar(bin_centers, hist_data, yerr = np.sqrt(hist_data), fmt='none', ecolor=plot_color[i], alpha = 0.4, elinewidth = 2)
            ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt = ".", color = plot_color[i], ecolor = plot_color[i], markersize = 16, elinewidth=2.5, markeredgecolor = 'black', markeredgewidth = 0.25)

            ax1.set_ylabel("Events", fontsize = 35)

            # Add thickness info to custom legend
            custom_handles_legend1.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = distance_label))
            custom_handles_legend2.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = f"$KS = {ks_stat:.3f},~p_{{KS}} = {ks_p_value:.3f}$"))



            ### Ratio Plots ....

            # Calculate the ratio, avoiding division by zero
            ratio = np.divide(hist_samples, hist_data, out=np.full_like(hist_data, fill_value= 0, dtype=float), where=hist_data != 0)

            inv_hist_data = np.divide(np.ones_like(hist_data, dtype=float), hist_data, out=np.zeros_like(hist_data, dtype=float), where=hist_data != 0)
            inv_hist_samples = np.divide(np.ones_like(hist_samples, dtype=float), hist_samples, out=np.zeros_like(hist_samples, dtype=float), where=hist_samples != 0)

            # get uncertainties with error propagation
            err = ratio * np.sqrt(inv_hist_data+ inv_hist_samples)

            ratio[ratio == 0] = -1




            bin_width = bin_centers[1] - bin_centers[0]

            # Determine offsets for ratios for better overview
            n_ratios = len(distance_bins)
            if n_ratios == 1:
                offsets = [0.0]
            elif n_ratios == 2:
                offsets = [-0.1 * bin_width, 0.1 * bin_width]
            else:
                offsets = np.linspace(-0.1 * bin_width, 0.1 * bin_width, n_ratios)


            # Plot the ratio on the 1. lower subplot
            ax2.errorbar(bin_centers + offsets[i], ratio, yerr = err, fmt = '.', color=plot_color[i], elinewidth= 2.5, markersize = 12, ecolor = plot_color[i], markeredgecolor = 'black', markeredgewidth = 0.25)
            ax2.axhline(1, color='black', linestyle='--')  # Reference line at y=1

            ax2.set_ylabel('')
            ymin, ymax = 0, 2
            ax2.set_ylim(ymin, ymax)  # Adjust y-limits for better view

            # Add arrows for values outside y-axis range
            for _, (xi, yi) in enumerate(zip(bin_centers, ratio)):
                if yi > ymax:
                    ax2.annotate('', xy=(xi, ymax - 0.1), xytext=(xi, ymax - 0.5),
                                arrowprops=dict(arrowstyle='->', color=plot_color[i], lw=2.5))

            # ylabel
            fig.text(0.02, 0.23, r'ParaFlow / MC', va='center', rotation='vertical', fontsize = 35)
                    
            # Plot the finer ratio on the lowest subplot
            ax3.errorbar(bin_centers + offsets[i], ratio, yerr = err, fmt = '.', color=plot_color[i], elinewidth= 2.5, markersize = 12, ecolor = plot_color[i], markeredgecolor = 'black', markeredgewidth = 0.25)
            ax3.axhline(1, color='black', linestyle='--')  # Reference line at y=1
            ax3.axhline(1.05, color='grey')  # Reference line at y=1.01
            ax3.axhline(0.95, color='grey')
            ax3.axhline(1.10, color='grey', ls = '--')
            ax3.axhline(0.90, color='grey', ls = '--')

            ax1.tick_params(axis='both', which='major', labelsize=30)
            ax2.tick_params(axis='both', which='major', labelsize=30)
            ax3.tick_params(axis='both', which='major', labelsize=30)

                
            # running max for getting the right y-axis limits
            running_max = max((max(hist_samples), max(hist_data), running_max))

        # first legend with info regarding thickness
        first_legend = ax1.legend(handles = custom_handles_legend1, loc = "upper right", fontsize = 27)
        ax1.add_artist(first_legend)

        # legend with statistical scores
        legend_properties = {'weight':'bold', 'size':27}
        second_legend = ax1.legend(handles = custom_handles_legend2, handlelength=0, loc = 'lower left', handletextpad=0, labelcolor = 'linecolor', fontsize = 27, prop = legend_properties, facecolor = "white", framealpha = 1, edgecolor = 'black', bbox_to_anchor=(0.18, 0.0075), frameon = True, fancybox = False)
        for item in second_legend.legend_handles:
            item.set_visible(False)
        frame = second_legend.get_frame()
        frame.set_edgecolor('black')  # Set the edge color to black
        frame.set_linewidth(2) 
        frame.set_facecolor('white') 
        ax1.add_artist(second_legend)

        # second legend to distinguish MC from FastSim
        ax1.errorbar(np.inf, np.inf, yerr = [1], color = "dimgrey", label = "ParaFlow", fmt=".", markersize = 15) # dump plot for legend
        ax1.errorbar(np.inf, np.inf, yerr = [1], color ="dimgrey", markersize = 0, elinewidth = 1, label = "MC", lw = 3.5) # dump plot for legend
        ax1.legend(loc = "upper left", fontsize = 30)
            
        # set axis limits
        ax1.set_ylim(0, running_max*1.5)
        if xlim: ax1.set_xlim(*xlim)
        if ylim: ax1.set_ylim(*ylim)

        # if we want a log scale
        if log_yscale: 
            ax1.set_yscale('log')


        ymin, ymax = 0.80, 1.20
        ax3.set_ylim(ymin, ymax)
        ax3.set_xlabel(x_label, fontsize = 35)

        fig.savefig(result_path + "bin_comparison_distance/comparison_" + filename + ".pdf", bbox_inches = 'tight')

        plt.close('all')



import numpy as np
import matplotlib.pyplot as plt



def plot_2d_parameterspace(data_array, samples_array, data_params, samples_params, title, result_path, colorbar_label, bins=30,  file_name = "shower_width_plot.png"):
    
    ''' Visualize the influence of parameters on shower variables with a 2D histograms. '''

    # Extract the two parameters for Samples
    samples_param1 = samples_params[:, 1] # thickness of material
    samples_param2 = samples_params[:, 2] # distance material to calorimeter
    

    # Create a 2D histogram with the mean of the property in each bin
    bin_means_samples, xedges, yedges, binnumber = binned_statistic_2d(
        samples_param1, samples_param2, samples_array, statistic='mean', bins=bins
    )

    # Repeat for MC
    data_param1 = data_params[:, 1]
    data_param2 = data_params[:, 2]
    
    # Create a 2D histogram with the mean of the property in each bin
    bin_means_data, xedges, yedges, binnumber = binned_statistic_2d(
        data_param1, data_param2, data_array, statistic='mean', bins=bins
    )
    
    # Get boundary values for colorbar
    vmin = min(np.min(bin_means_samples), np.min(bin_means_data))
    vmax = max(np.max(bin_means_samples), np.max(bin_means_data))


    ## Plotting for FastSim
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, width_ratios = [3.5,4.5,4.5], figsize = (30,8), sharey = True, sharex=True)


    fig.suptitle("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}~-~Parameter~Space~Visualisation}}$", x = 0.1, y = 1.05, c="black", fontsize = 45, ha = 'left', va = 'top')

    im = ax1.imshow(bin_means_samples.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis', vmin = vmin , vmax = vmax)
    ax1.set_title("ParaFlow", c="navy", fontweight = 'bold', fontsize = 40)
    ax1.set_ylabel(r"Distance Iron-Detector [cm]", fontsize = 40)

    ## Plotting for MC
    im = ax2.imshow(bin_means_data.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis', vmin = vmin , vmax = vmax)
    ax2.set_title("MC", c="navy", fontweight = 'bold', fontsize = 40)
    cbar = fig.colorbar(im, ax=ax2, extend = 'min')
    cbar.set_label(colorbar_label, fontsize = 35)
    cbar.ax.tick_params(labelsize=30)

    
   
    ### Plot difference of heatmaps
    difference = ((bin_means_data.T - bin_means_samples.T) / bin_means_data.T) * 100

    ax3.set_title("Difference", c="navy", fontweight = 'bold', fontsize = 40)

    # Samples
    im = ax3.imshow(difference, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=-5, vmax=5, cmap='bwr')
    ax3.set_xlabel(r"Thickness Iron [$X_0$]", fontsize = 40)
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label('(MC - ParaFlow)/MC [%]', fontsize = 35)
    cbar.ax.tick_params(labelsize=30)

    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax2.tick_params(axis='both', which='major', labelsize=40)
    ax3.tick_params(axis='both', which='major', labelsize=40)

     # Save the plot to file
    fig.savefig(result_path + file_name + ".pdf", bbox_inches = 'tight')
    plt.close()
