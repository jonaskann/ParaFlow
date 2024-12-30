import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep


def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',size=20,**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="right",pad=-25, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.57, -0.10 ),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)

plt.rcdefaults()
plt.style.use([hep.style.ROOT])


bins = np.array([2,4,6,8,10,12,14])
AUCs_bins = np.array([0.7100, 0.6022, 0.5985,0.5860, 0.5744, 0.5832, 0.5956])

transforms = np.array([2,4,6,8,10,12,14])
AUC_transforms = np.array([0.7354, 0.6287, 0.6020, 0.5815, 0.5744, 0.5890, 0.5824])

fig, ax  = plt.subplots(1, 1, figsize = (10,6))

# ax.plot(bins, AUCs_bins, c = "red", alpha = 0.1, lw = 1)
bin_plot = ax.scatter(bins, AUCs_bins, c = 'darkmagenta', marker="d", label = "RQS Bins (MADE Blocks = 10)", s = 100)
transform_plot = ax.scatter(transforms, AUC_transforms, c = '#13785a', marker='p', label = "MADE Blocks (RQS Bins = 10)", s = 100)


ax.legend(loc='best', fontsize = 20)

ax.tick_params(axis='both', which='major', labelsize=15)


ax.set_ylabel("AUC", fontsize = 20)
multicolor_ylabel(ax = ax, list_of_strings=["# MADE blocks  ", '# RQS bins'], list_of_colors=['#13785a', 'darkmagenta'], anchorpad=0)

ax.grid('--', alpha = 0.7)

fig.savefig("hyperparameter_classifier.pdf", bbox_inches = 'tight')
