
def one_col_figsize(aspect_ratio):
    return 3.26772, 3.26772/aspect_ratio


def two_col_figsize(aspect_ratio):
    return 4.724, 4.724/aspect_ratio

display_figure_instead = False
def savefig(fig, fname):
    if display_figure_instead:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(f'/home/liam/Copernicus_LaTeX_Package/figures/{fname}', dpi=300, bbox_inches='tight', pad_inches=0)