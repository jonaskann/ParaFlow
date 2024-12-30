

import numpy as np

from scipy.stats import ks_2samp, binned_statistic_2d

import matplotlib
from matplotlib import pyplot as plt
import mplhep as hep

plt.rcdefaults()
plt.style.use([hep.style.ROOT])

#plt.rcParams['figure.constrained_layout.use'] = True





def get_binedges(bin_centers):

    bin_edges = np.concatenate((
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[1:] + bin_centers[:-1]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]
    ))

    return bin_edges


def chi2(hist1, hist2, n_bins):

        mask = (hist1 == 0) & (hist2 == 0)
        hist1 = hist1
        hist2 = hist2
        error = np.sqrt(hist1 + hist2) # poisson error on both histograms
        error[mask] = 1
        return np.sum((((hist1 - hist2) ** 2) / (error)**2))/n_bins


def plot_histogram(data_samples, data_geant4, bin_centers, particle_type, plot_color, error_color, title, x_label, y_label, filename, result_path, data_conditions, sample_conditions, xlim = None, ylim = None, calc_std = False, detector_noise_level = None, shielding_range = None, distance_range = None, bin_comparison_shielding = False, bin_comparison_distance = False, log_yscale = False):

    import numpy as np

    bin_edges = get_binedges(bin_centers=bin_centers)

    hist_samples, _ = np.histogram(data_samples, bins = bin_edges)
    hist_data, _            = np.histogram(data_geant4, bins = bin_edges)


    ####### metrics #####
    chi2_value = chi2(hist1=hist_data, hist2=hist_samples, n_bins=len(bin_centers))

    # Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(data_samples, data_geant4, alternative = 'two-sided')


    # plotting
    


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex = True, gridspec_kw={'hspace': 0.1, 'height_ratios': [7, 1.5, 1.5]})

    particle_title = r"\gamma" if particle_type == 'photon' else r"$\pi_0$"

    if (shielding_range == (0.5,1.5)) and (distance_range == (50,90)):
        condition_title = "" 
    
    elif (shielding_range != (0.5,1.5)):
        condition_title = f"$\\bf{{d \\in ({shielding_range[0]},{shielding_range[1]})~X_0}}$" 
    else:
        condition_title = f"$\\bf{{b \\in ({distance_range[0]},{distance_range[1]})~cm}}$"

    ax1.set_title("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$" , c="black", fontsize = 35, loc = 'left')
    ax1.set_title(condition_title, c = "dimgrey", fontsize =25, loc = 'right')

    option1 = True

    if option1:

        ax1.fill_between(bin_centers, hist_data, step='mid', color=plot_color, alpha = 0.7, label = r"MC", edgecolor = 'black')

        uncertainties_data = np.sqrt(hist_data)
        # Add the uncertainty grid patches
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

        ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt='.', markersize = 12, markeredgecolor ='black', markerfacecolor = 'black', ecolor='black', elinewidth = 2, label = "Fast Simulation")


    elif not option1:
        ax1.fill_between(bin_centers, hist_data, step='mid', color=plot_color, alpha = 0.2, label = r"MC")
        ax1.errorbar(bin_centers, hist_data, yerr = np.sqrt(hist_data), fmt='none', ecolor=plot_color, alpha = 0.3, elinewidth = 2)
        ax1.step(bin_centers, hist_samples, where='mid', color=plot_color, lw = 2.5, label = "Fast Simulation")
        ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt='', markersize = 12, ecolor=plot_color, elinewidth = 2)


    if xlim: ax1.set_xlim(*xlim)
    ax1.set_ylim(0, max(max(hist_samples), max(hist_data))*1.4)
    if ylim: ax1.set_ylim(*ylim)
    ax1.set_ylabel(y_label, fontsize = 35)
    if log_yscale: 
        ax1.set_ylim(100, max(max(hist_samples), max(hist_data))*1.2)
        if ylim: ax1.set_ylim(*ylim)
        ax1.set_yscale('log')
    ax1.legend(loc = 'upper left', fontsize = 30)
    # We change the fontsize of minor ticks label 
    ax1.tick_params(axis='both', which='major', labelsize=30)

    # relevant values of statistics
    text = f'$\chi^2 / n_{{\mathrm{{bins}}}} = {chi2_value:.2f}$\n$KS = {ks_stat:.4f}$\n$p_{{KS}} = {ks_p_value:.4f}$'

    # add a text box with the statistic values
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

    # # used parameters
    # text = f'$\\bf{{Detector Conditions}}$\n$d \\in ({shielding_range[0]},{shielding_range[1]})~X_0$\n$b \\in ({distance_range[0]},{distance_range[1]})~cm$'

    # # add a text box with the statistic values
    # props = dict(boxstyle="square, pad=0.3", edgecolor="none", facecolor="none", alpha = 0.2)
    # ax1.text(
    #     0.400, 0.96,  # Coordinates relative to the axes
    #     text, 
    #     transform=ax1.transAxes, 
    #     fontsize=30, 
    #     bbox=props,
    #     verticalalignment='top',  # Align box to the top
    #     horizontalalignment='left',  # Align box to the right
    # )
    

    if calc_std:
         
         ax1.text(
            plt.gca().get_xlim()[0] + abs(plt.gca().get_xlim()[0] - plt.gca().get_xlim()[1]) * 0.05,  # x-coordinate
            max(max(hist_samples), max(hist_data)) * 0.95 if not log_yscale else max(max(hist_samples), max(hist_data)) / 20,
            f'Expected $\sigma = {12*detector_noise_level:.1f}$\n$\hat{{\sigma}}_{{\mathrm{{Geant4}}}} = {np.std(data_geant4, ddof=1):.1f}$\n$\hat{{\sigma}}_{{\mathrm{{Samples}}}} = {np.std(data_samples, ddof=1):.1f}$', 
            fontsize=20, 
            color=plot_color,
            fontweight = "bold",
            ha='left',  # Align text to the right
            va='top'
        )



    # Calculate the ratio, avoiding division by zero
    ratio = np.divide(hist_data, hist_samples, out=np.full_like(hist_data, fill_value= 0, dtype=float), where=hist_samples != 0)

    inv_hist_data = np.divide(np.ones_like(hist_data, dtype=float), hist_data, out=np.zeros_like(hist_data, dtype=float), where=hist_data != 0)
    inv_hist_samples = np.divide(np.ones_like(hist_samples, dtype=float), hist_samples, out=np.zeros_like(hist_samples, dtype=float), where=hist_samples != 0)

    if option1:
        
        err_data = uncertainties_data * inv_hist_samples

        err_samples = hist_data * (inv_hist_samples)**(3/2)

        ratio[ratio == 0] = -1

        # Plot the ratio on the lower subplot
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

        fig.text(0.02, 0.23, r'MC / FastSim', va='center', rotation='vertical', fontsize = 35)

        ax3.axhline(1, color='black', linestyle='--')  # Reference line at y=1
        ax3.axhline(1.05, color='grey')  # Reference line at y=1.01
        ax3.axhline(0.95, color='grey')
        ax3.axhline(1.10, color='grey', ls = '--')
        ax3.axhline(0.90, color='grey', ls = '--')

        # Plot the finer ratio on the lowest subplot
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
        


    elif not option1:    

        err = ratio * np.sqrt(inv_hist_data+ inv_hist_samples)

        ratio[ratio == 0] = -1

        # Plot the ratio on the lower subplot
        ax2.errorbar(bin_centers, ratio, yerr = err, fmt = '.', color=plot_color, elinewidth= 2.5, markersize = 12, ecolor = plot_color)
        ax2.axhline(1, color='black', linestyle='--')  # Reference line at y=1

        ax2.set_ylabel('')
        ymin, ymax = 0, 2
        ax2.set_ylim(ymin, ymax)  # Adjust y-limits for better view

        # Add arrows for values outside y-axis range
        for i, (xi, yi) in enumerate(zip(bin_centers, ratio)):
            if yi > ymax:
                ax2.annotate('', xy=(xi, ymax - 0.1), xytext=(xi, ymax - 0.5),
                            arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

        fig.text(0.02, 0.23, r'MC / FastSim', va='center', rotation='vertical', fontsize = 35)
                
        # Plot the finer ratio on the lowest subplot
        ax3.errorbar(bin_centers, ratio, yerr = err, fmt = '.', color=plot_color, elinewidth= 2.5, markersize = 12, ecolor = plot_color)
        ax3.axhline(1, color='black', linestyle='--')  # Reference line at y=1
        ax3.axhline(1.05, color='grey')  # Reference line at y=1.01
        ax3.axhline(0.95, color='grey')
        ax3.axhline(1.10, color='grey', ls = '--')
        ax3.axhline(0.90, color='grey', ls = '--')

    ax3.set_xlabel(x_label, fontsize = 35)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax3.tick_params(axis='both', which='major', labelsize=30)


    ax3.set_ylabel('')
    ymin, ymax = 0.80, 1.20
    ax3.set_ylim(ymin, ymax)  # Adjust y-limits for better view
    
    fig.savefig(result_path + filename + '_' + particle_type + ".pdf", bbox_inches='tight')


    #########################################################################################################
    #        Add another plot, where we compare the histograms for the different shielding bins             #
    #########################################################################################################

    if (bin_comparison_shielding):

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex = True, gridspec_kw={'hspace': 0.1, 'height_ratios': [7, 1.5, 1.5]})

        particle_title = r"$\gamma$" if particle_type == 'photon' else r"$\pi_0$"


        shielding_bins = [(0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5)]
        plot_color = ['orange', 'red', 'royalblue', 'darkmagenta']
        hatches = ['/', '\\', '--', '||']
        custom_handles_legend1 = []
        custom_handles_legend2 = []

        # running max for y-limit
        running_max = 0

        for i, shielding_range in enumerate(shielding_bins):

            shield_label = f"$d \\in ({shielding_range[0]},{shielding_range[1]})~X_0$" 


            mask_samples = (sample_conditions[:,1] < shielding_range[1]) & (sample_conditions[:,1] > shielding_range[0])
            mask_data    = (data_conditions[:,1] < shielding_range[1]) & (data_conditions[:,1] > shielding_range[0])

            data_samples_filtered = data_samples[mask_samples]
            data_geant4_filtered  = data_geant4[mask_data]

            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(data_samples_filtered, data_geant4_filtered, alternative = 'two-sided')

            hist_samples, _ = np.histogram(data_samples_filtered, bins = bin_edges)
            hist_data, _            = np.histogram(data_geant4_filtered, bins = bin_edges)


            ax1.set_title("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}}}$", c="black", fontsize = 35, loc = 'left', pad = 15)

            ax1.errorbar(bin_centers, hist_data, yerr = np.sqrt(hist_data), fmt='none', ecolor=plot_color[i], alpha = 0.4, elinewidth = 2)
            ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt = ".", color = plot_color[i], ecolor = plot_color[i], markersize = 16, elinewidth=2.5, markeredgecolor = 'black', markeredgewidth = 0.25)
            ax1.fill_between(bin_centers, hist_data, step='mid', color=plot_color[i], edgecolor = plot_color[i], lw = 0, alpha = 0.025)
            ax1.step(bin_centers, hist_data, where='mid', color=plot_color[i], lw = 3.5, alpha = 0.6)

            # Add thickness info to custom legend
            custom_handles_legend1.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = shield_label))
            custom_handles_legend2.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = f"$KS = {ks_stat:.3f},~p_{{KS}} = {ks_p_value:.3f}$"))

            ax1.set_ylabel("Events", fontsize = 35)


            # Ratio Plot


            # Calculate the ratio, avoiding division by zero
            ratio = np.divide(hist_data, hist_samples, out=np.full_like(hist_data, fill_value= 0, dtype=float), where=hist_samples != 0)

            inv_hist_data = np.divide(np.ones_like(hist_data, dtype=float), hist_data, out=np.zeros_like(hist_data, dtype=float), where=hist_data != 0)
            inv_hist_samples = np.divide(np.ones_like(hist_samples, dtype=float), hist_samples, out=np.zeros_like(hist_samples, dtype=float), where=hist_samples != 0)


            err = ratio * np.sqrt(inv_hist_data+ inv_hist_samples)

            ratio[ratio == 0] = -1

            bin_width = bin_centers[1] - bin_centers[0]

            # Determine offsets for ratios
            n_ratios = len(shielding_bins)
            if n_ratios == 1:
                offsets = [0.0]
            elif n_ratios == 2:
                offsets = [-0.1 * bin_width, 0.1 * bin_width]
            else:
                offsets = np.linspace(-0.1 * bin_width, 0.1 * bin_width, n_ratios)

            # Plot the ratio on the lower subplot
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

            fig.text(0.02, 0.23, r'MC / FastSim', va='center', rotation='vertical', fontsize = 35)
                    
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
        location = 'center right' if "R9" not in title else 'center left'
        legend_properties = {'weight':'bold', 'size':27}
        second_legend = ax1.legend(handles = custom_handles_legend2, handlelength=0, handletextpad=0, loc = 'best', labelcolor = 'linecolor', fontsize = 27, prop = legend_properties, facecolor = "white", framealpha = 1, edgecolor = 'black', bbox_to_anchor=(0, 0.35, 1, 0.28), frameon = True, fancybox = False)
        # Customize the frame around the legend
        frame = second_legend.get_frame()
        frame.set_edgecolor('black')  # Set the edge color to black
        frame.set_linewidth(2) 
        frame.set_facecolor('white')  # Ensure the background is not transparent

        for item in second_legend.legend_handles:
            item.set_visible(False)
        ax1.add_artist(second_legend)


        # second legend to distinguish MC from FastSim
        ax1.errorbar(np.inf, np.inf, yerr = [1], color = "dimgrey", label = "Fast Simulation", fmt=".", markersize = 15) # dump plot for legend
        ax1.errorbar(np.inf, np.inf, yerr = [1], color ="dimgrey", markersize = 0, elinewidth = 1, label = "MC", lw = 3.5) # dump plot for legend
        ax1.legend(loc = "upper left", fontsize = 30)
        
        

        # set axis limits
        ax1.set_ylim(0, running_max*1.5)
        if xlim: ax1.set_xlim(*xlim)
        if ylim: ax1.set_ylim(*ylim)

        # log_yscale if needed
        if log_yscale: 
            ax1.set_ylim(100, max(max(hist_samples), max(hist_data))*1.5)
            if ylim: ax1.set_ylim(*ylim)
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
        


        fig.savefig(result_path + "bin_comparison_shielding/comparison_" + filename + ".pdf", bbox_inches = 'tight')

        plt.close('all')

    if (bin_comparison_distance):

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex = True, gridspec_kw={'hspace': 0.1, 'height_ratios': [7, 1.5, 1.5]})

        particle_title = r"$\gamma$" if particle_type == 'photon' else r"$\pi_0$"

        distance_bins = [(50, 60), (60, 70), (70, 80), (80, 90)]
        plot_color = ['orange', 'red', 'royalblue', 'darkmagenta']
        hatches = ['/', '\\', '--', '||']
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

            ax1.errorbar(bin_centers, hist_data, yerr = np.sqrt(hist_data), fmt='none', ecolor=plot_color[i], alpha = 0.4, elinewidth = 2)
            ax1.errorbar(bin_centers, hist_samples, yerr = np.sqrt(hist_samples), fmt = ".", color = plot_color[i], ecolor = plot_color[i], markersize = 16, elinewidth=2.5, markeredgecolor = 'black', markeredgewidth = 0.25)
            ax1.fill_between(bin_centers, hist_data, step='mid', color=plot_color[i], edgecolor = plot_color[i], lw = 0, alpha = 0.025)
            ax1.step(bin_centers, hist_data, where='mid', color=plot_color[i], lw = 3.5, alpha = 0.6)

            ax1.set_ylabel("Events", fontsize = 35)

            # Add thickness info to custom legend
            custom_handles_legend1.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = distance_label))
            custom_handles_legend2.append(matplotlib.patches.Patch(facecolor = plot_color[i], label = f"$KS = {ks_stat:.3f},~p_{{KS}} = {ks_p_value:.3f}$"))



            ### Ratio Plot 1

            # Calculate the ratio, avoiding division by zero
            ratio = np.divide(hist_data, hist_samples, out=np.full_like(hist_data, fill_value= 0, dtype=float), where=hist_samples != 0)

            inv_hist_data = np.divide(np.ones_like(hist_data, dtype=float), hist_data, out=np.zeros_like(hist_data, dtype=float), where=hist_data != 0)
            inv_hist_samples = np.divide(np.ones_like(hist_samples, dtype=float), hist_samples, out=np.zeros_like(hist_samples, dtype=float), where=hist_samples != 0)


            err = ratio * np.sqrt(inv_hist_data+ inv_hist_samples)

            ratio[ratio == 0] = -1

            bin_width = bin_centers[1] - bin_centers[0]

            # Determine offsets for ratios
            n_ratios = len(distance_bins)
            if n_ratios == 1:
                offsets = [0.0]
            elif n_ratios == 2:
                offsets = [-0.1 * bin_width, 0.1 * bin_width]
            else:
                offsets = np.linspace(-0.1 * bin_width, 0.1 * bin_width, n_ratios)

            # Plot the ratio on the lower subplot
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

            fig.text(0.02, 0.23, r'MC / FastSim', va='center', rotation='vertical', fontsize = 35)
                    
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
        location = 'center right' if "R9" not in title else 'center left'
        legend_properties = {'weight':'bold', 'size':27}
        second_legend = ax1.legend(handles = custom_handles_legend2, handlelength=0, handletextpad=0, loc = 'best', labelcolor = 'linecolor', fontsize = 27, prop = legend_properties, facecolor = "white", framealpha = 1, edgecolor = 'black', bbox_to_anchor=(0, 0.35, 1, 0.28), frameon = True, fancybox = False)
        for item in second_legend.legend_handles:
            item.set_visible(False)
        frame = second_legend.get_frame()
        frame.set_edgecolor('black')  # Set the edge color to black
        frame.set_linewidth(2) 
        frame.set_facecolor('white') 
        ax1.add_artist(second_legend)

        # second legend to distinguish MC from FastSim
        ax1.errorbar(np.inf, np.inf, yerr = [1], color = "dimgrey", label = "Fast Simulation", fmt=".", markersize = 15) # dump plot for legend
        ax1.errorbar(np.inf, np.inf, yerr = [1], color ="dimgrey", markersize = 0, elinewidth = 1, label = "MC", lw = 3.5) # dump plot for legend
        ax1.legend(loc = "upper left", fontsize = 30)
            
        # set axis limits
        ax1.set_ylim(0, running_max*1.5)
        if xlim: ax1.set_xlim(*xlim)
        if ylim: ax1.set_ylim(*ylim)

        # if we want a log scale
        if log_yscale: 
            ax1.set_ylim(100, max(max(hist_samples), max(hist_data))*1.5)
            if ylim: ax1.set_ylim(*ylim)
            ax1.set_yscale('log')


        ymin, ymax = 0.80, 1.20
        ax3.set_ylim(ymin, ymax)
        ax3.set_xlabel(x_label, fontsize = 35)

        fig.savefig(result_path + "bin_comparison_distance/comparison_" + filename + ".pdf", bbox_inches = 'tight')

        plt.close('all')



import numpy as np
import matplotlib.pyplot as plt



def plot_2d_parameterspace(data_array, samples_array, data_params, samples_params, title, result_path, colorbar_label, bins=30,  file_name = "shower_width_plot.png"):
    """
    Visualize the influence of parameters on shower width with a 2D image plot.
    
    Parameters:
    - data: numpy array of shape (n,), the calculated shower widths.
    - params: numpy array of shape (n, 2), the two parameters for each calculation.
    - bins: int or tuple of int, number of bins for the x and y axes (default: 50).
    - output_file: str, file path to save the resulting plot (default: "shower_width_plot.png").
    """

    #### Get 2D parameter visualisation ####

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
    
    vmin = min(np.min(bin_means_samples), np.min(bin_means_data))
    vmax = max(np.max(bin_means_samples), np.max(bin_means_data))


    particle_title = r"$\gamma$" 


    ## Plotting for FastSim
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, width_ratios = [3.5,4.5,4.5], figsize = (30,8), sharey = True, sharex=True)


    fig.suptitle("$\\bf{{ParaFlow}}$  "  + f"$\\it{{{title}~-~Parameter~Space~Visualisation}}$", x = 0.1, y = 1.05, c="black", fontsize = 45, ha = 'left', va = 'top')

    im = ax1.imshow(bin_means_samples.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis', vmin = vmin , vmax = vmax)
    ax1.set_title("FastSim", c="navy", fontweight = 'bold', fontsize = 40)
    # ax1.set_xlabel(r"Thickness Iron [$X_0$]")
    ax1.set_ylabel(r"Distance Iron-Detector [cm]", fontsize = 40)
    # cbar = fig.colorbar(im, ax=ax, extend = 'min')
    # cbar.set_label(colorbar_label)

    
    

    ## Plotting for MC
    # ax = fig.add_subplot(1,3,2, width_ratios = [2,3,3])
    im = ax2.imshow(bin_means_data.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis', vmin = vmin , vmax = vmax)
    ax2.set_title("MC", c="navy", fontweight = 'bold', fontsize = 40)
    # ax2.set_xlabel(r"Thickness Iron [$X_0$]")
    # ax2.set_ylabel(r"Distance Iron-Detector [cm]")
    cbar = fig.colorbar(im, ax=ax2, extend = 'min')
    cbar.set_label(colorbar_label, fontsize = 35)
    cbar.ax.tick_params(labelsize=30)

    
   
    ### Plot difference of Heatmaps

    difference = ((bin_means_data.T - bin_means_samples.T) / bin_means_data.T) * 100


    # ax = fig.add_subplot(1,3,3, width_ratios = [2,3,3])

    ax3.set_title("Difference", c="navy", fontweight = 'bold', fontsize = 40)

    # Samples
    im = ax3.imshow(difference, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=-7.5, vmax=7.5, cmap='bwr')
    ax3.set_xlabel(r"Thickness Iron [$X_0$]", fontsize = 40)
    # ax3.set_ylabel(r"Distance Iron-Detector [cm]")
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label('(MC - FastSim)/MC [%]', fontsize = 35)
    cbar.ax.tick_params(labelsize=30)

    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax2.tick_params(axis='both', which='major', labelsize=40)
    ax3.tick_params(axis='both', which='major', labelsize=40)

     # Save the plot to file
    fig.savefig(result_path + file_name + ".pdf", bbox_inches = 'tight')
    plt.close()



# Example usage:
# data = np.random.rand(1000)  # Example shower width data
# params = np.random.rand(1000, 2)  # Example parameters
# plot_shower_width(data, params)


##############################################################


import matplotlib.pyplot as plt 
import mplhep as hep
plt.style.use([hep.style.ROOT])
import os
import numpy as np
from hist import Hist
from copy import deepcopy


def plot_histograms_CMS(
    hists,
    process_labels,
    output_filename = "./plot.pdf", 
    axis_labels=("x-axis","Events"), 
    normalize=False,
    linestyle="solid",
    log=False,
    include_flow=False,
    data=False,
    lumi=None,
    return_figure=False,
    ax = None,
):

    if include_flow:
        hists = [include_overflow_underflow(hist) for hist in hists]
    if normalize:
        integrals = [_hist.sum().value for _hist in hists]
        hists = [_hist / integral for _hist, integral in zip(hists, integrals)]
    binning = hists[0].to_numpy()[1]
    values = [_hist.values() for _hist in hists]
    uncertainties = [np.sqrt(_hist.variances()) for _hist in hists]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,9))
    else:
        # meant to be used with fig created outside of this function
        fig = None
    hep.histplot(
        values,
        label=process_labels,
        bins=binning,
        linewidth=3,
        yerr=uncertainties,
        ax=ax,
        linestyle=linestyle
    )

    ax.margins(y=0.15)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.05*ax.get_ylim()[1])
        ax.tick_params(labelsize=22)
    ax.set_xlabel(axis_labels[0], fontsize=26)
    ax.set_ylabel(axis_labels[1], fontsize=26)
    ax.tick_params(labelsize=24)

    # adjust legend line width
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
        handle.get_children()[0].set_linewidth(3)  # Line
        handle.get_children()[1].set_linewidth(3)  # Caplines
        # handle.get_children()[2].set_linewidth(3)  # Bars
        new_handles.append(handle)
    ncols = 1 if len(hists) < 4 else 2
    ax.legend(handles=new_handles, labels=labels, loc="upper right", fontsize=24, ncols=ncols)#, handlelength=3)

    # in case the figure should be modified later on, use return_figure=True
    if not return_figure:
        if not os.path.exists(output_filename.replace(output_filename.split("/")[-1], "")):
            os.makedirs(output_filename.replace(output_filename.split("/")[-1], ""))
        plt.tight_layout()
        fig.savefig(output_filename)
        plt.close()
        return
    else:
        return fig, ax



def include_overflow_underflow(hist):

    # do not change in place:
    new_hist = deepcopy(hist)
    bin_contents = new_hist.view(flow=True)
    overflow, underflow = bin_contents[-1], bin_contents[0]
    n_bins = len(bin_contents) - 2 
    # print(new_hist.view(flow=True))
    new_hist[0] += underflow
    new_hist[n_bins-1] += overflow
    # print(new_hist.view(flow=True))
    return new_hist

