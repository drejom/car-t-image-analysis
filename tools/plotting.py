import matplotlib
matplotlib.use("TkAgg")
import tools.general_tools as gt

import os
from scipy.interpolate import griddata
import SimpleITK as sitk
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



#==== HELPERS


def interpolate_over_grid(x, y, u, v, n=100, return_coords='linspace', method='cubic'):
    """
    Interpolates 2D vector field data with (values u, v, at positions x, y) over grid with n nodes.
    - return_coords='linspace' -> x_return, y_return each contain n values
    - return_coords='meshgrid' -> x_return, y_return each contain nxn values
    """
    # from:
    # https://stackoverflow.com/questions/33637693/how-to-use-streamplot-function-when-1d-data-of-x-coordinate-y-coordinate-x-vel
    nx = ny = n
    #-- (N, 2) arrays of input x,y coords and u,v values
    pts  = np.vstack((x, y)).T
    vals = np.vstack((u, v)).T
    #-- the new x and y coordinates for the grid, which will correspond to the
    #   columns and rows of u and v respectively
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    #-- an (nx * ny, 2) array of x,y coordinates to interpolate at
    ipts = np.vstack(a.ravel() for a in np.meshgrid(xi,yi)).T
    #-- an (nx * ny, 2) array of interpolated u, v values
    ivals = griddata(pts, vals, ipts, method=method)
    #-- reshape interpolated u,v values into (nx, ny) arrays
    ui, vi = ivals.T
    ui.shape = vi.shape = (ny, nx)
    if return_coords=='meshgrid':
        x_return = ipts[:,0]
        y_return = ipts[:,1]
    elif return_coords=='linspace':
        x_return = xi
        y_return = yi
    return x_return, y_return, ui, vi


def get_ranges_colormap(values, range=None, cmap='gist_earth', norm=None, norm_ref=None, n_cmap_levels=None,
                        **kwargs):
    """
    Parses commandline arguments to construct boundaries, colormap and norm.
    """
    values_flat = np.ndarray.flatten(values)
    if range is None:
        min_ = min(values_flat)
        max_ = max(values_flat)
    else:
        min_ = range[0]
        max_ = range[1]

    if type(cmap) == str:
        if n_cmap_levels:
            cmap_  = plt.cm.get_cmap(cmap, n_cmap_levels)
        else:
            cmap_ = plt.cm.get_cmap(cmap)
    else:
        cmap_ = cmap

    if norm_ref is None:
        norm_ref = (min_+max_)/2.

    if norm is None:
        norm = MidpointNormalize(midpoint=norm_ref, vmin=min_, vmax=max_)
    return min_, max_, cmap_, norm


def exclude_from_data(data, min_f, max_f,
                      exclude_below=None, exclude_above=None, exclude_min_max=False, exclude_around=None,
                      min_max_eps=0.00001, data_type='standard'):
    """
    Produces a boolean mask basked with True value when any or several selection criteria is met
    - data_type='standard'      ->  True/False value for each entry in 'data' array.
    - data_type='triangulation' ->  True/False value for each triangle (data triple) in 'data' array.
    If no criterium applies, the returned mask contains False values only
    """
    if not exclude_above:
        exclude_above = max_f # either from data value range, or from 'range' attibute -> get_ranges_colormap()
    if not exclude_below:
        exclude_below = min_f

    mask_list = []
    if exclude_min_max:
        if data_type=='standard':
            mask1 = np.ma.make_mask(np.where(data > exclude_above + min_max_eps, 1, 0))
            mask2 = np.ma.make_mask(np.where(data < exclude_below - min_max_eps, 1, 0))
        elif data_type=='triangulation':
            mask1 = np.logical_or.reduce((np.where(data > exclude_above+min_max_eps, 1, 0).T))
            mask2 = np.logical_or.reduce((np.where(data < exclude_below-min_max_eps, 1, 0).T))
        mask_min_max = np.logical_or(mask1, mask2)
        mask_list.append(mask_min_max)

    if type(exclude_around)==list:
        ref = exclude_around[0]
        eps = exclude_around[1]
        if data_type=='standard':
            mask1 = np.ma.make_mask(np.where(data > ref - eps, 1, 0))
            mask2 = np.ma.make_mask(np.where(data < ref - eps, 1, 0))
        elif data_type=='triangulation':
            mask_below = np.logical_or.reduce((np.where(data > ref - eps, 1, 0).T))
            mask_above = np.logical_or.reduce((np.where(data < ref - eps, 1, 0).T))
        mask_above_below = np.logical_or(mask_below, mask_above)
        mask_list.append(mask_above_below)

    if len(mask_list)>1:
        mask = mask_list[0]
        for i in range(1, len(mask_list)-1):
            mask = np.logical_or(mask,mask_list[i] )
    elif len(mask_list)==1:
        mask = mask_list[0]
    else:
        if data_type == 'standard':
            mask = np.ma.make_mask(np.zeros(data.shape))
        else:
            mask = np.logical_or.reduce(np.zeros(data.shape).T)
    return mask



def add_colorbar(fig, ax, img_handle, label=None):
    divider = make_axes_locatable(ax)
    cbax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img_handle, cax=cbax)
    cbar.ax.get_yaxis().labelpad = 17
    #cbar.ax.tick_params(labelsize=base_font_size - 2)
    if not label == None:
        cbar.ax.set_ylabel(label, rotation=270) #, fontsize=base_font_size)
    return cbax


#==== PLOTTING SUBFUNCTIONS


#--- SITK IMAGE
def plot_sitk_image(ax, image, segmentation=None, contour=False,
                    range=None, cmap='gist_earth', norm=None, norm_ref=0, n_cmap_levels=None,
                    exclude_below=None, exclude_above=None, exclude_min_max=None, exclude_around=None,
                    label_alpha=1,
                    origin='lower',
                    **kwargs):
    #-- Convert Image Types
    img_type = sitk.sitkUInt8
    #image_rec = sitk.Cast(sitk.RescaleIntensity(image), img_type)
    image_rec = sitk.RescaleIntensity(image)
    #-- Segmentation
    if segmentation:
        image_label_rec = sitk.Cast(segmentation, img_type)
        if contour:
            img_ol = sitk.LabelOverlay(image_rec, sitk.LabelContour(image_label_rec), opacity=label_alpha)
        else:
            img_ol = sitk.LabelOverlay(image_rec, image_label_rec, opacity=label_alpha)
    else:
        img_ol = image_rec
    #-- Prepare Figure with image
    nda = sitk.GetArrayFromImage(img_ol)
    min_f, max_f, colormap, norm = get_ranges_colormap(nda,
                                                       range=range, cmap=cmap, norm=norm, norm_ref=norm_ref,
                                                       n_cmap_levels=n_cmap_levels)

    mask = exclude_from_data(nda, min_f, max_f,
                             exclude_below=exclude_below, exclude_above=exclude_above,
                             exclude_min_max=exclude_min_max, exclude_around=exclude_around,
                             data_type='standard')
    nda[mask] = np.nan

    img_origin  = img_ol.GetOrigin()
    img_spacing = img_ol.GetSpacing()
    img_size    = img_ol.GetSize()
    # extent -> (left, right, bottom, top)
    extent = (img_origin[0], img_origin[0] + img_size[0] * img_spacing[0],
              img_origin[1] + img_size[1] * img_spacing[1], img_origin[1])
    plot = ax.imshow(nda, interpolation=None, extent=extent, origin=origin,
                     cmap=colormap, norm=norm, vmin=min_f, vmax=max_f)
    return plot


#--- MAIN PLOT FUNCTION
def plot(plot_object_list, dpi=100, plot_range=None, margin=0.02, cbarwidth=0.05,
         save_path=None, show=True, xlabel='x position [mm]', ylabel='y position [mm]', **kwargs):
    """
    Each element in plot_object_list is a dictionary containing of:
        'object' : the object to be plotted
        'param1' : one plot specific parameter
        ...
    unless a 'zorder' argument is specified, elements are plotted in the order of occurrence in the list
    """

    # -- create Figure
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')

    # -- if an image is provided, the axes will be oriented as in the image,
    #   otherwise use xlim, ylim for fixing display orientation
    # -- for discussion about imshow/extent https://matplotlib.org/tutorials/intermediate/imshow_extent.html
    if plot_range:
        ax.set_xlim(*plot_range[0:2])
        ax.set_ylim(*plot_range[2:4])


    if 'title' in kwargs:
        fig.suptitle(kwargs.pop('title'))
    #-- Check if only a single plot_object has been provided, if so, transform to dict
    if not type(plot_object_list)==list:
        plot_object_dict = {}
        plot_object_dict['object'] = plot_object_list
        plot_object_dict.update(kwargs)
        plot_object_list = [plot_object_dict]
    # -- Iterate through plot objects
    cbar_ax_list = [ax]
    for plot_object_dict_orig in plot_object_list:
        plot_object_dict = plot_object_dict_orig.copy()
        plot_object = plot_object_dict.pop('object')
        # check for 'color'
        if 'color' in plot_object_dict:
            color = plot_object_dict['color']
        else:
            color=False
        cbar = False
        if ('cbar_label' in plot_object_dict) and not color:
            cbar_label = plot_object_dict.pop('cbar_label')
            cbar=True

        #-- use global kwargs as reference and overwrite with settings that are specific to this plot_object
        params = kwargs.copy()
        params.update(plot_object_dict)
        #-- plot
        if type(plot_object)==sitk.Image:
            plot = plot_sitk_image(ax, plot_object, **params)
        else:
            print("The plot_object is of type '%s' -- not supported."%(type(plot_object)))
            raise Exception

        if cbar:
            cbax = add_colorbar(fig, cbar_ax_list[0], plot, cbar_label)
            cbar_ax_list.append(cbax)
            #fig.subplots_adjust(left=margin, bottom=margin, right=1 - margin - cbarwidth, top=1 - 2 * margin)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if save_path:
        gt.ensure_dir_exists(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches='tight',  dpi=dpi)
        print("-- saved figure to '%s'"%save_path)

    if show:
        plt.show()
    else:
        plt.close()

