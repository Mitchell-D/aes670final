""" """

from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
import pickle as pkl
from pprint import pprint as ppt
import numpy as np

from aes670hw2 import guitools as gt
from aes670hw2 import enhance as enh
from aes670hw2 import geo_plot as gp
from aes670hw2 import TextFormat as TFmt
from aes670hw2 import MOD021KM
from aes670hw2 import Recipe

'''
""" Subgrid 1 """
buf_dir = Path("./buffer")
fig_dir = Path("./figures")
data_dir = Path("./data")
target_latlon = (19.588, -155.51)
target_time = dt(year=2017, month=6, day=13, hour=20, minute=54)
satellite = "terra"
region_width, region_height = 640, 512
hdf_path = data_dir.joinpath(f"subgrid.hdf")
subgrid_pkl = data_dir.joinpath(f"subgrid.pkl")
kmeans_pkl = data_dir.joinpath(f"subgrid_kmeans.pkl")
thresh_pkl = data_dir.joinpath(f"subgrid_thresh.pkl")
'''

""" Subgrid 2 """
buf_dir = Path("./buffer")
fig_dir = Path("./figures2")
data_dir = Path("./data")
#target_latlon = (1.112, 35.428)
target_latlon = (32.416, 32.987)
#target_time = dt(year=2023, month=3, day=24, hour=7, minute=48)
target_time = dt(year=2019, month=5, day=10, hour=8, minute=29)
#satellite = "terra"
satellite = "terra"
region_width, region_height = 640, 512
hdf_path = data_dir.joinpath(f"subgrid2.hdf")
subgrid_pkl = data_dir.joinpath(f"subgrid2.pkl")
kmeans_pkl = data_dir.joinpath(f"subgrid2_kmeans.pkl")
thresh_pkl = data_dir.joinpath(f"subgrid2_thresh.pkl")
samples_pkl = data_dir.joinpath(f"subgrid2_samples.pkl")
mlc_pkl = data_dir.joinpath(f"subgrid2_mlc.pkl")
masks_pkl = data_dir.joinpath(f"subgrid2_masks.pkl")


token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"
l1b_bands = (
    3,  # 459-479nm blue
    10, # 483-493nm teal/blue
    4,  # 545-565nm green
    1,  # 620-670nm near-red
    2,  # 841-876nm NDWI / land boundaries
    16, # 862-877nm NIR / aerosol distinction
    19, # 916-965nm H2O absorption
    5,  # 1230-1250nm optical depth
    26, # 1360-1390nm cirrus band
    6,  # 1628-1652nm snow/ice band
    7,  # 2106-2155nm cloud particle size
    20, # 3660-3840nm SWIR
    21, # 3929-3989 another SWIR
    27, # 6535-6895nm Upper H2O absorption
    28, # 7175-7475nm Lower H2O absorption
    29, # 8400-8700nm Infrared cloud phase, emissivity diff 11-8.5um
    31, # 10780-11280nm clean LWIR
    32, # 11770-12270nm less clean LWIR
    33, # 14085-14385nm dirty LWIR
    )

def restore_subgrid(pkl_path):
    """ Verify that the region pkl exists. If not, re-load it. """
    if not pkl_path.exists():
        """
        Ask the user whether to download a new HDF, which is useful when settings
        change or the HDF is deleted.
        """
        if input("Download new hdf? (Y/n) ").lower()=="y":
            tmp_path = MOD021KM.download_granule(
                    data_dir=data_dir,
                    raw_token = token,
                    target_latlon=target_latlon,
                    satellite=satellite,
                    target_time=target_time,
                    day_only=True,
                    debug=debug,
                    )
            print(f"Downloaded l1b: {tmp_path.name}")
            tmp_path.rename(hdf_path)
        """ Load the requested bands from the current hdf_path granule """
        M = MOD021KM.from_hdf(hdf_path, l1b_bands)
        subgrid = M.get_subgrid(target_latlon, region_height, region_width,
                          from_center=True, boundary_error=False)
        subgrid.make_pkl(pkl_path)
    else:
        """ If it exists, load the subgrid pkl as a MOD021KM object. """
        subgrid = MOD021KM.from_pkl(pkl_path)
    return subgrid

def do_kmeans(subgrid, km_bands:list, km_pkl:Path, fig_dir:Path,
              km_class_count:int, batch_label:str, get_new:bool=False,
              plot_classes:bool=False, colors:list=None, title:str=""):
    if km_pkl.exists():
        km_arr = pkl.load(km_pkl.open("rb"))
    else:
        km_arr = {}
    if get_new:
        new_batch = subgrid.get_kmeans(
                labels=km_bands, class_count=km_class_count,
                return_as_ints=True, debug=True)
        km_arr[batch_label] = (new_batch, km_bands)
        pkl.dump(km_arr, km_pkl.open("wb"))

    if plot_classes:
        gp.plot_classes(
                class_array=km_arr[batch_label][0],
                class_labels=[f"KM{i+1}" for i in range(km_class_count)],
                colors=colors,
                plot_spec={
                    "title":title
                    },
                fig_path=fig_dir.joinpath(
                    #f"class/kmeans_all_{km_class_count}c.png"),
                    f"class/kmeans_{batch_label}_{km_class_count}c.png"),
                show=False,
                )

def get_raw_images(subgrid, fig_dir, hsv_params:dict={}):
    """ Get a grayscale and HSV scalar raster for each band """
    print(TFmt.YELLOW("Generating grayscale rasters for all bands"))
    for b in l1b_bands:
        gp.generate_raw_image(enh.norm_to_uint(subgrid.data(b),256,np.uint8),
                              fig_dir.joinpath(Path(f"scalar/gs_{b:02}.png")))
        gp.generate_raw_image(
                enh.norm_to_uint(gt.scal_to_rgb(subgrid.data(b),**hsv_params),
                                 256, np.uint8),
                fig_dir.joinpath(Path(f"scalar/hsv_{b:02}.png")))
        if subgrid.info(b)["is_reflective"]:
            gp.generate_raw_image(
                    enh.norm_to_uint(gt.scal_to_rgb(
                        subgrid.raw_reflectance(b),**hsv_params),256,np.uint8),
                    fig_dir.joinpath(Path(f"scalar/rawref_{b:02}.png")))

def get_sg2_surface_masks(subgrid, fig_dir, thresh_pkl):
    """ """
    def _update_thresh_pkl(mask_key, mask):
        # Load already-selected masks back from the pkl
        if not thresh_pkl.exists():
            stored_masks = {}
        else:
            stored_masks = pkl.load(thresh_pkl.open("rb"))
        stored_masks[mask_key] = mask
        # Load the new masks back into the pkl
        pkl.dump(stored_masks, thresh_pkl.open("wb"))

    """ Cirrus mask is a simple lower-bound on the cirrus band """
    mask_key = "ice_cloud"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select lower bound for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("DCP", lower=True, upper=False,
                                     choose_rgb_params=True,
                                     rgb_match=True)
        # Invert the mask so that ice cloud pixels are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ mid-clouds are discerned with the day cloud-phase RGB """
    mask_key = "water_cloud"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("DCP", lower=True, upper=True,
                                     choose_rgb_params=True, rgb_match=True)
        land_mask, _ = subgrid.get_mask(31, upper=True, rgb_type="CUSTOM",
                                     show=True, rgb_match=True)
        mask = np.logical_or(mask, land_mask)
        rgb[np.where(np.dstack([mask for i in range(3)]))] = 0
        # Invert the mask so that water cloud pixels are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ arid ground is discerned with my custom RGB """
    mask_key = "arid"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("CUSTOM", lower=True, upper=True,
                                     choose_rgb_params=True, rgb_match=True)
        # Invert the mask so that arid is True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ vegetation is discerned with my custom RGB """
    mask_key = "vegetation"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("CUSTOM", lower=True, upper=True,
                                     choose_rgb_params=True, rgb_match=True)
        # Invert the mask so that vegetated are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ Water is discerned with the CUSTOM RGB and a lower bound on band 31 """
    mask_key = "water"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("CUSTOM", upper=True, lower=True,
                                     show=False, rgb_match=True)
        ndwi_mask, _ = subgrid.get_mask("ndwi", upper=True, lower=True,
                                     use_hsv=True,show=False, rgb_match=True)
        # Bounds on LWIR
        land_mask, _ = subgrid.get_mask(31, lower=True, upper=True,
                                        rgb_type="CUSTOM", show=False,
                                        rgb_match=True)
        # Water is most ambiguous, so negate all other masks
        masks_dict = pkl.load(thresh_pkl.open("rb"))
        #mask_labels, all_thresh_masks = tuple(zip(*masks_dict.items()))
        #not_water_mask = np.any(np.dstack(all_thresh_masks),axis=2)
        #not_water_mask = np.logical_or(not_water_mask, mask)
        #water_mask = np.logical_not(not_water_mask)
        mask = np.logical_or(mask, land_mask)
        #mask = np.logical_and(mask, cloud_mask)
        mask = np.logical_or(mask, ndwi_mask)
        rgb[np.where(np.dstack([mask for i in range(3)]))] = 0

        # Invert the mask so that water are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

def get_sg1_surface_masks(subgrid, fig_dir, thresh_pkl):
    """ """
    def _update_thresh_pkl(mask_key, mask):
        # Load already-selected masks back from the pkl
        if not thresh_pkl.exists():
            stored_masks = {}
        else:
            stored_masks = pkl.load(thresh_pkl.open("rb"))
        stored_masks[mask_key] = mask
        # Load the new masks back into the pkl
        pkl.dump(stored_masks, thresh_pkl.open("wb"))

    """ Cirrus mask is a simple lower-bound on the cirrus band """
    mask_key = "ice_cloud"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select lower bound for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask(26, lower=True, upper=False,
                                     choose_rgb_params=True,
                                     rgb_type="DUST", rgb_match=True)
        # Invert the mask so that ice cloud pixels are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ mid-clouds are discerned with the day cloud-phase RGB """
    mask_key = "water_cloud"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("DCP", lower=True, upper=True,
                                     choose_rgb_params=True, rgb_match=True)
        # Invert the mask so that water cloud pixels are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ vegetation is discerned with my custom RGB """
    mask_key = "igneous"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE("new igneous", bold=True))
        oig_mask, rgb = subgrid.get_mask(
                16, lower=True, upper=True, rgb_type="TCeq",
                choose_rgb_params=True,rgb_match=True)
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("CUSTOM", lower=True, upper=True,
                                     choose_rgb_params=True, rgb_match=True)
        # Invert the mask so that igneous are True
        inv_mask = np.logical_not(mask)
        #_update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))
        old_igneous_mask = np.logical_and(inv_mask, oig_mask)
        new_igneous_mask = np.logical_and(inv_mask, np.logical_not(oig_mask))
        _update_thresh_pkl("old_igneous", old_igneous_mask)
        _update_thresh_pkl("new_igneous", new_igneous_mask)

    """ mid-clouds are discerned with the day cloud-phase RGB """
    mask_key = "vegetation"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask("CUSTOM", lower=True, upper=True,
                                     choose_rgb_params=True, rgb_match=True)
        # Invert the mask so that vegetated are True
        inv_mask = np.logical_not(mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

    """ mid-clouds are discerned with the day cloud-phase RGB """
    mask_key = "water"
    instr = TFmt.RED(f"Update {mask_key} mask? ",bright=True) + \
            TFmt.WHITE("(Y/n): ")
    if input(instr).lower() == "y":
        print(TFmt.RED(f"Select upper and lower bounds for", bright=True),
              TFmt.WHITE(mask_key, bold=True))
        mask, rgb = subgrid.get_mask(31, upper=True, lower=True, show=True,
                                     use_hsv=True, rgb_match=True)
        # Water is most ambiguous, so negate all other masks
        masks_dict = pkl.load(thresh_pkl.open("rb"))
        mask_labels, all_thresh_masks = tuple(zip(*masks_dict.items()))
        not_water_mask = np.any(np.dstack(all_thresh_masks),axis=2)
        not_water_mask = np.logical_or(not_water_mask, mask)
        water_mask = np.logical_not(not_water_mask)
        rgb[np.where(np.dstack([not_water_mask for i in range(3)]))] = 0

        # Invert the mask so that water are True
        inv_mask = np.logical_not(not_water_mask)
        _update_thresh_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

def do_threshold_spectral(subgrid, thresh_pkl):
    """
    Generate spectral distributions for pixels that belong to each mask.

    This method does not update the threshold pkl
    """
    mask_labels, masks = tuple(zip(*pkl.load(thresh_pkl.open("rb")).items()))
    subgrid.spectral_analysis(
            masks=masks,
            mask_labels=mask_labels,
            show=True,
            plot_spec={
                "yscale":"logit",
                "yrange":(0,.8),
                "alpha":.3,
                "title":"Threshold-masked surface type reflectance responses",
                "xlabel":"MODIS reflectance channels",
                "ylabel":"TOA Bidirectional Reflectance Factor",
                })
    subgrid.spectral_analysis(
            array_labels = [b for b in subgrid.bands
                            if not subgrid.info(b)["is_reflective"]],
            masks=masks,
            mask_labels=mask_labels,
            show=True,
            plot_spec={
                #"yscale":"logit",
                "yrange":(220,330),
                "alpha":.3,
                "title":"Threshold-masked surface type emissive responses",
                "xlabel":"MODIS emissive channels",
                "ylabel":"TOA Brightness Temperature",
                })
    """
    Generate spectral distributions for pixels that DON'T belong to each mask.
    """
    subgrid.spectral_analysis(
            masks=[ np.logical_not(m) for m in masks],
            mask_labels=["NOT "+l for l in mask_labels],
            show=True,
            plot_spec={
                "yscale":"logit",
                "yrange":(0,.4),
                "alpha":.3,
                "title":"Threshold-masked surface type reflectance responses",
                "xlabel":"MODIS reflectance channels",
                "ylabel":"TOA Bidirectional Reflectance Factor",
                })
    subgrid.spectral_analysis(
            array_labels = [b for b in subgrid.bands
                            if not subgrid.info(b)["is_reflective"]],
            masks=[ np.logical_not(m) for m in masks],
            mask_labels=["NOT "+l for l in mask_labels],
            show=True,
            plot_spec={
                #"yscale":"logit",
                "yrange":(220,330),
                "alpha":.3,
                "title":"Threshold-masked surface type emissive responses",
                "xlabel":"MODIS emissive channels",
                "ylabel":"TOA Brightness Temperature",
                })

def get_rgbs(subgrid, fig_dir, gamma_scale:int=2, choose=False):
    for r in subgrid.rgb_recipes.keys():
        print(TFmt.GREEN("Choose RGB gamma for ")+TFmt.WHITE(r, bold=True))
        rgb = subgrid.get_rgb(r, choose_gamma=choose, gamma_scale=gamma_scale)
        gp.generate_raw_image(rgb, fig_dir.joinpath(f"rgbs/rgb_{r}.png"))

if __name__=="__main__":
    """ Settings """
    debug = True
    colors = {
        "vegetation":[.16,.65,.29],
        "water":[.10,.55,1],
        "water_cloud":[1,1,1],
        "ice_cloud":[.5,1,1],
        "arid":[.82,.76,.56],
        "uncertain":[0,0,0]}
    hsv_params = {"hue_range":(.6,0),"sat_range":(.6,.9),"val_range":(.6,.6)}

    subgrid = restore_subgrid(subgrid_pkl)
    #get_rgbs(subgrid, fig_dir, choose=True, gamma_scale=4)
    #get_sg2_surface_masks(subgrid, fig_dir, thresh_pkl)
    #do_threshold_spectral(subgrid, thresh_pkl)
    #get_raw_images(subgrid, fig_dir, hsv_params)

    '''
    """ Get gamma-enhanced and HSV-mapped RGB renders of band 28 and NDVI """
    cbar_array = np.stack((np.linspace(0,1,512) for i in range(64)), axis=1).T
    gp.generate_raw_image(gt.scal_to_rgb(cbar_array,**hsv_params),
                          fig_dir.joinpath("scalar/cbar.png"))
    gp.generate_raw_image(
            gt.scal_to_rgb(subgrid.data(28, choose_gamma=True),**hsv_params),
            fig_dir.joinpath("rgbs/gamma_hsv_28.png"))
    gp.generate_raw_image(
            gt.scal_to_rgb(subgrid.data(
                "ndvi",choose_gamma=True),**hsv_params),
            fig_dir.joinpath("rgbs/gamma_ndvi.png"))
    '''

    '''
    """ Add a recipe for my histogram-equalized custom RGB (except NDVI) """
    subgrid.add_recipe(
            "Band 31, Norm",
            Recipe((31,),lambda a: enh.linear_gamma_stretch(
                enh.histogram_equalize(a,1024)[0])))
    subgrid.add_recipe(
            "NDVI Norm",
            Recipe(("ndvi",),lambda a: enh.linear_gamma_stretch(
                np.clip(a,0,1))))
    subgrid.add_recipe(
            "Inv. Band 29, Norm",
            Recipe((29,31),
                   lambda a,b: enh.linear_gamma_stretch(b/a)))
    subgrid.add_rgb_recipe(
            "CUSTOMeq",
            Recipe(
                args=["Band 31, Norm","NDVI Norm","Inv. Band 29, Norm"],
                func=lambda a,b,c:enh.linear_gamma_stretch(np.dstack([a,b,c])))
            )
    '''

    '''
    """
    Generate my custom RGB and HSV-mapped scalar incredients, as well as a
    brightness level histogram for the ingredients.
    """
    # Get an equalized custom RGB
    custom_rgb = subgrid.get_rgb(
            "CUSTOMeq",choose_gamma=False,choose_contrast=False,gamma_scale=3)
    # Histogram-match the custom RGB to normal distributions
    custom_rgb = subgrid.histogram_match(
            custom_rgb,
            gt.get_normal_rgb(
                *subgrid.shape, means=(.5,.3,.5), stdevs=(.2,.2,.3)),
            nbins=1024,
            show=True)
    # Add the custom RGB as new RGB data
    subgrid.add_rgb_data("CUSTOMhist", enh.linear_gamma_stretch(custom_rgb))
    # Do histogram analysis on the gamma and histogram enhanced RGB
    subgrid.rgb_histogram_analysis(
            rgb_label="CUSTOMhistgamma",nbins=1024,show=True,
            plot_spec={"colors":[[1,0,0],[0,1,0],[0,0,1]], "line_width":.8,
                       "yrange":(0,.02), "xrange":(0,1)},
            fig_path=fig_dir.joinpath("spectra/custom_matched.png"))
    # Gamma-enhance the histogram-matched RGB

    """ Gamma-enhance the histogram-matched RGB """
    custom_rgb = subgrid.get_rgb("CUSTOMhist", choose_gamma=True,gamma_scale=2)
    # Add the gamma-enhanced RGB as new data.
    subgrid.add_rgb_data("CUSTOMhistgamma", custom_rgb)
    # Do histogram analysis on the gamma and histogram enhanced RGB
    subgrid.rgb_histogram_analysis(
            rgb_label="CUSTOMhistgamma",nbins=1024,show=True,
            plot_spec={"colors":[[1,0,0],[0,1,0],[0,0,1]], "line_width":.8,
                       "yrange":(0,.02), "xrange":(0,1)},
            fig_path=fig_dir.joinpath("spectra/custom_matched.png"))
    gp.generate_raw_image(subgrid.get_rgb("CUSTOMhist"),
                          fig_dir.joinpath("rgbs/rgb_CUSTOMhist.png"))
    gp.generate_raw_image(subgrid.get_rgb("CUSTOMhistgamma"),
                          fig_dir.joinpath("rgbs/rgb_CUSTOMhistgamma.png"))

    gp.generate_raw_image(
            gt.scal_to_rgb(custom_rgb[:,:,0], **hsv_params),
            fig_dir.joinpath("rgbs/rgb_CUSTOMhistgamma_RED.png"))
    gp.generate_raw_image(
            gt.scal_to_rgb(custom_rgb[:,:,1], **hsv_params),
            fig_dir.joinpath("rgbs/rgb_CUSTOMhistgamma_GREEN.png"))
    gp.generate_raw_image(
            gt.scal_to_rgb(custom_rgb[:,:,2], **hsv_params),
            fig_dir.joinpath("rgbs/rgb_CUSTOMhistgamma_BLUE.png"))
    '''

    '''
    """ Do K-means classification """
    #km_bands=(3,4,1,2,7), # ref only
    #km_bands=(20,21,29,31,32) # thermal-only
    #km_bands=(1,2,3,4,5,19,21,26,28,29,31) # Batch 2
    #km_bands=(1,2,3,4,6,"ndvi",21,26,28,29,31) # Batch 3
    #km_bands=(1,10,20,21,29,31,32) # Batch 4
    #km_bands=(3,4,1,2,7,29,31,32) # Batch 5
    #km_bands=(3,4,1,2,7,26,29,31,32) # Batch 6
    #km_bands=(3,4,1,2,7,26,29,31,32,33) # Batch 7 w/ cirrus band
    #km_bands=(3,4,1,2,7,19,29,31,32,33) # Batch 8 w/ NIR abs.
    km_bands=(1,2,3,4,5,19,20,26,28,29,31) # Batch 9; batch 2 w/20 not 21
    #km_bands=(3,4,1,2,7,19,29,31,32,33) # Batch 10; batch 8
    km_class_count = 10
    do_kmeans(
            subgrid,
            km_bands=km_bands,
            batch_label=f"batch10c{km_class_count}",
            km_class_count=km_class_count,
            km_pkl=kmeans_pkl,
            fig_dir=fig_dir,
            title=f"Results ({km_class_count} class)",
            get_new=False,
            plot_classes=False,
            )
    '''

    '''
    """ Merge K-means classes from batch 9 10 class run """
    km_dict = pkl.load(kmeans_pkl.open("rb"))
    km = km_dict["batch9c10"][0]
    new_km = np.zeros_like(km)
    new_km[np.where(km==0)] = 0 # vegetation
    new_km[np.where(km==8)] = 0 # vegetation
    new_km[np.where(km==1)] = 1 # water
    new_km[np.where(km==6)] = 2 # mid clouds
    new_km[np.where(km==7)] = 2 # mid clouds
    new_km[np.where(km==2)] = 3 # cirrus
    new_km[np.where(km==9)] = 3 # cirrus
    new_km[np.where(km==3)] = 3 # cirrus
    new_km[np.where(km==4)] = 4 # arid
    new_km[np.where(km==5)] = 4 # arid

    #km_dict["merged"] = (new_km, km_bands)
    km_dict["merged2"] = (new_km, km_bands)
    pkl.dump(km_dict, kmeans_pkl.open("wb"))
    # Subjective class ID order of batch9c10 classification
    km_order = ["vegetation", "water", "water_cloud", "ice_cloud", "arid"]
    km_colors = [ colors[l] for l in km_order ]
    do_kmeans(
            subgrid,
            km_bands=km_bands,
            # Be very careful with label keys! Don't overwrites something.
            #batch_label="merged",
            batch_label="merged2",
            km_class_count=5,
            km_pkl=kmeans_pkl,
            fig_dir=fig_dir,
            title=f"Merged Batch 9 Classes",
            get_new=False,
            plot_classes=True,
            colors=km_colors,
            )
    '''

    '''
    """ Do maximum-likelihood classification """
    samples = 400
    thresh = .9
    mlc_bands = (1,2,3,4,5,19,20,26,28,29,31) # Batch 9
    #mlc_bands = (1,2,3,4,5,19,20,28,29,31,32) # Batch 10
    #mlc_bands = None
    #mlc_batch = "all-thresh"
    #mlc_batch = "b9-thresh"
    #mlc_batch = "b10-thresh"
    use_km = True
    #mlc_batch = "all-km"
    mlc_batch = "b9-km"
    title = f"Maximum-likelihood Classification Results ({mlc_batch})"
    if use_km:
        """ Get samples from K-means results """
        km_order = ["vegetation", "water", "water_cloud", "ice_cloud", "arid"]
        km_colors = [ colors[l] for l in km_order ]
        if thresh:
            km_colors.append(colors["uncertain"])
        #km_ints_merged, km_bands = pkl.load(kmeans_pkl.open("rb"))["merged"]
        km_ints_merged, km_bands = pkl.load(kmeans_pkl.open("rb"))["merged2"]
        # For each K-means class, get a list of pixel samples
        mlc_samples = [MOD021KM.mask_to_idx(M, samples)
                      for M in MOD021KM.ints_to_masks(km_ints_merged)]
        mlc_labels = [f"KM{i+1}" for i in range(len(mlc_samples))]
        tmp_mlc_dict = {mlc_labels[i]:mlc_samples[i]
                       for i in range(len(mlc_labels))}
    else:
        """ Get samples from my threshold masks """
        # Convert masks to #samples index tuples
        mlc_labels, mlc_samples = zip(*[
                (c,MOD021KM.mask_to_idx(M, samples))
                for c,M in pkl.load(thresh_pkl.open("rb")).items()])
        tmp_mlc_dict = {mlc_labels[i]:mlc_samples[i]
                       for i in range(len(mlc_labels))}

    """ Run MLC and plot classes. """
    mlc_ints, mlc_labels = subgrid.get_mlc(tmp_mlc_dict,mlc_bands,thresh)
    gp.plot_classes(
            class_array=mlc_ints,
            class_labels=mlc_labels,
            colors= km_colors if use_km else [colors[l] for l in mlc_labels],
            plot_spec={ "title":title },
            fig_path=fig_dir.joinpath(
                #f"class/kmeans_all_{km_class_count}c.png"),
                f"class/mlc_{mlc_batch}_{len(mlc_labels)}c.png"),
            show=False,
            )
    # Convert the sample indeces back into boolean masks for storage
    sample_labels, sample_masks = tuple(zip(*[
            (k,MOD021KM.idx_to_mask(X,subgrid.shape))
            for k,X in tmp_mlc_dict.items()]))

    # Update dictionaries with full MLC masks and masks for just the samples.
    mlc_dict = pkl.load(mlc_pkl.open("rb"))
    samples_dict = pkl.load(samples_pkl.open("rb"))
    # Overwrite the current MLC batch in both dictionaries
    samples_dict[mlc_batch] = (sample_masks, sample_labels)
    mlc_dict[mlc_batch] = (mlc_ints, mlc_labels)
    pkl.dump(samples_dict, samples_pkl.open("wb"))
    pkl.dump(mlc_dict, mlc_pkl.open("wb"))
    '''

    '''
    """ Classification done. collect all masks into one pkl """
    # Returns the number of elements in common between masks
    confusion = lambda m1,m2: np.count_nonzero(np.logical_and(m1,m2))

    km_ints, km_bands = pkl.load(kmeans_pkl.open("rb"))["merged2"]
    km_masks = MOD021KM.ints_to_masks(km_ints)
    km_labels = [f"KM{i+1}" for i in range(np.amax(km_ints)+1)]

    thresh_labels, thresh_masks = zip(*pkl.load(thresh_pkl.open("rb")).items())

    mlc_run = "b9-km"
    mlc_ints, mlc_km_labels = pkl.load(mlc_pkl.open("rb"))[mlc_run]
    mlc_km_masks = MOD021KM.ints_to_masks(mlc_ints)
    samples_km_masks, samples_km_labels = pkl.load(
            samples_pkl.open("rb"))[mlc_run]

    mlc_run = "b9-thresh"
    mlc_ints, mlc_thresh_labels = pkl.load(mlc_pkl.open("rb"))[mlc_run]
    mlc_thresh_masks = MOD021KM.ints_to_masks(mlc_ints)
    samples_thresh_masks, samples_thresh_labels = pkl.load(
            samples_pkl.open("rb"))[mlc_run]

    """
    IMPORTANT! This is the final step of the classification procedure. The
    masks pickle is a dictionary mapping each classification type's ID string
    to a tuple of labels for that class, which are mapped to a 2-tuple
    containing equal-sized lists of string labels for each class and (M,N)
    boolean masks with the corresponding class set to True. These masks
    correspond to one full run of the procedure. For my final project,
    I used samples from the following bands for all classification:
    b9_bands = (1,2,3,4,5,19,20,26,28,29,31)
    """
    if input(TFmt.RED(
        "Are you SURE you want to overwrite the merged class pickle? ") + \
                TFm.WHITE("(Y/n)",bold=True)).lower() == "y":
        pkl.dump({
            "thresh":(thresh_labels, thresh_masks),
            "km":(km_labels, km_masks), # merged
            "samples_km":(samples_km_labels, samples_km_masks),
            "samples_thresh":(samples_thresh_labels, samples_thresh_masks),
            "mlc_km":(mlc_km_labels, mlc_km_masks),
            "mlc_thresh":(mlc_thresh_labels, mlc_thresh_masks),
            }, masks_pkl.open("wb"))
    '''

    """ -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- """
    """         Prepare for analysis of classifications and samples         """
    """ -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- -=- """
    b9_bands=(1,2,3,4,5,19,20,26,28,29,31) # Batch 9; batch 2 w/20 not 21
    b9_ref_bands = [b for b in b9_bands if subgrid.info(b)["is_reflective"]]
    b9_temp_bands = [b for b in b9_bands if b not in b9_ref_bands]
    b9_ref = np.dstack([subgrid.data(b) for b in b9_ref_bands])
    b9_temp = np.dstack([subgrid.data(b) for b in b9_temp_bands])
    masks_dict = pkl.load(masks_pkl.open("rb"))


    '''
    """ Generate pixel masks for every class in every classification run """
    base_rgb = "TC"
    for rk in masks_dict.keys():
        rl, rm = masks_dict[rk]
        print(rk,rl)
        for i in range(len(rl)):
            pass
            gp.generate_raw_image(MOD021KM.mask_to_color(subgrid.get_rgb(
                base_rgb), rm[i], radius=1),fig_dir.joinpath(
                    f"colormask/cmask_{rk}_{rl[i]}_{base_rgb}.png"))
    '''


    #'''
    """
    Generate spectral distribution graphics for the selected classification run
    """
    run_key = "mlc_km"
    type_str = "Maximum-likelihood classification results from K-means samples"

    print(TFmt.GREEN(f"Selected run {run_key} from {list(masks_dict.keys())}"))
    run_cats, run_masks = masks_dict[run_key]
    subgrid.spectral_analysis(
            b9_ref_bands, run_cats, run_masks,
            plot_spec={
                "title":f"Batch 9 {type_str} Reflectance Spectral Response",
                "xlabel":"MODIS band wavelengths",
                "ylabel":"Bidirectional Reflectance Factor",
                },
            fig_path=fig_dir.joinpath(f"spectra/ref_{run_key}_b9.png")
            )
    subgrid.spectral_analysis(
            b9_temp_bands, run_cats, run_masks,
            plot_spec={
                "title":f"Batch 9 {type_str} Temperature Spectral Response",
                "xlabel":"MODIS band wavelengths",
                "ylabel":"Brightness Temperature (K)",
                "yrange":(240,330),
                },
            fig_path=fig_dir.joinpath(f"spectra/temp_{run_key}_b9")
            )
    all_ref_bands = [b for b in subgrid.bands
                     if subgrid.info(b)["is_reflective"]]
    all_temp_bands = [b for b in subgrid.bands if b not in all_ref_bands]
    subgrid.spectral_analysis(
            mask_labels=run_cats, masks=run_masks,
            plot_spec={
                "title":f"All-band {type_str} Reflectance Spectral Response",
                "xlabel":"MODIS band wavelengths",
                "ylabel":"Bidirectional Reflectance Factor",
                "yrange":(0,1),
                },
            fig_path=fig_dir.joinpath(f"spectra/ref_{run_key}_all.png")
            )
    subgrid.spectral_analysis(
            all_temp_bands, run_cats, run_masks,
            plot_spec={
                "title":f"All-band {type_str} Temperature Spectral Response",
                "xlabel":"MODIS band wavelengths",
                "ylabel":"Brightness Temperature (K)",
                "yrange":(240,330),
                },
            fig_path=fig_dir.joinpath(f"spectra/temp_{run_key}_all")
            )
    #'''

    '''
    """
    Print information on area, covariance, standard deviation, and covariance
    """
    """ Print pixel count and area info for each run """
    #gt.quick_render(gt.scal_to_rgb(subgrid.data("sza")))
    #print(TFmt.WHITE(
    #    f"\n -----( AREAS )----- ", bold=True))

    print(np.amin(subgrid.data("vza")), np.amax(subgrid.data("vza")))
    print(subgrid.data("vza").shape)
    print(TFmt.GREEN(f"Total area: {subgrid.area()}", bold=True))
    gp.generate_raw_image(enh.linear_gamma_stretch(subgrid.data("vza")),
                          fig_dir.joinpath("scalar/vza.png"))
    km_runs = []
    normal_runs = []
    for run in masks_dict.keys():
        run_labels, masks = masks_dict[run]
        run_str  = "\\hline\n"
        run_str += run.replace("_"," ")+" Pixels & " + \
                " & ".join([str(int(np.count_nonzero(M))) for M in masks])
        run_str +=  "\\\\\n"+run.replace("_"," ")+" Areas & " + \
                " & ".join([str(int(subgrid.area(M))) for M in masks])+"\\\\"
        if "km" in run:
            km_header = "& "+" & ".join(run_labels) + "\\\\"
            km_runs.append(run_str)
        else:
            normal_header = "& "+" & ".join(
                    [l.replace("_"," ") for l in run_labels]) + "\\\\"
            normal_runs.append(run_str)
    print("\\begin{figure}[h!]\n\\centering")
    print("\\begin{tabular}{c|" + "".join(
        ["c" for i in range(len(run_labels)+1)]) + "}")
    print(normal_header)
    for s in normal_runs:
        print(s)
    print("\\end{tabular}")
    print("\\begin{tabular}{c|" + "".join(
        ["c" for i in range(len(run_labels)+1)]) + "}")
    print(km_header)
    for s in km_runs:
        print(s)
    print("\\end{tabular}")
    print("\\caption{Pixel class areas and counts}")
    print("\\label{pixel_areas}\n\\end{figure}")

    '''

    '''
    """ Get mean, stdev, and covariance tables for each class in a run """
    # runs: thresh, km, samples_thresh, mlc_thresh, samples_km, mlc_km
    run = "mlc_km"
    run_labels, run_masks = masks_dict[run]
    """ Get a table for reflectance bands """
    print("\n\\clearpage\n")
    print("\\begin{figure}[h!]\n\\centering")
    print("\\begin{tabular}{C|C|C|" + "".join(
        ["C" for i in range(len(b9_ref_bands))]) + "}\n")
    ref_uncertain = None
    col_str = "\lambda & \mu & \sigma & \multicolumn{"
    col_str += str(len(b9_ref_bands))+"}{c}{" + \
            "Reflectance Covariance $(\\times10^{4})$} \\\\\n"
    print(col_str)
    for i in range(len(run_labels)):
        ref_px = b9_ref[np.where(run_masks[i])].T
        ref_px = [ref_px[i,:] for i in range(ref_px.shape[0])]
        # Sort the bands by their center wavelengths
        b9_ref_wl = [subgrid.info(b)["ctr_wl"] for b in b9_ref_bands]
        _, b9_ref_bands, ref_px = zip(*sorted(
            zip(b9_ref_wl, b9_ref_bands, ref_px), key=lambda t: t[0]))
        ref_px = np.squeeze(np.dstack(ref_px)).T
        # Get the mean and stdev vectors and covariance matrix
        ref_avg = np.average(ref_px, axis=1)
        ref_cov = np.cov(ref_px)
        ref_stdev = np.std(ref_px, axis=1)
        header_str = "\\hline\n"
        header_str += "\\multicolumn{"+str(len(b9_ref_bands)+3)+"}{c}{" + \
                run_labels[i].replace("_"," ")+"} \\\\\n\\hline"
        rows = ""
        # Make a row for each band
        for j in range(len(b9_ref_bands)):
            row_str = f"{subgrid.info(b9_ref_bands[j])['ctr_wl']:.3f} & "
            row_str += f"{ref_avg[j]:.3f} & {ref_stdev[j]:.3f} & "
            row_str += " & ".join(map(
                lambda s:f"{s:.1f}",list(10000*ref_cov[j,:]))) + " \\\\\n"
            rows += row_str
        if run_labels[i]=="uncertain":
            ref_uncertain = (header_str, col_str, rows)
        else:
            print(header_str)
            print(rows)
    print("\\end{tabular}")
    print("\\caption{"+run.replace("_", " ")+" reflectance statistics}")
    print("\\label{"+run+"_ref_stats}\n\\end{figure}")

    """ Get a table for thermal bands """
    print("\n\\clearpage\n")
    print("\\begin{figure}[h!]\n\\centering")
    print("\\begin{tabular}{C|C|C|" + "".join(
        ["C" for i in range(len(b9_temp_bands))]) + "}\n")
    temp_uncertain = None
    col_str = "\lambda & \mu & \sigma & \multicolumn{"
    col_str += str(len(b9_temp_bands))+"}{c}{" + \
            "Brightness Temp. Covariance $(\\times10^{2})$} \\\\\n"
    print(col_str)
    for i in range(len(run_labels)):
        temp_px = b9_temp[np.where(run_masks[i])].T
        temp_px = [temp_px[i,:] for i in range(temp_px.shape[0])]
        # Sort the bands by their center wavelengths
        b9_temp_wl = [subgrid.info(b)["ctr_wl"] for b in b9_temp_bands]
        _, b9_temp_bands, temp_px = zip(*sorted(
            zip(b9_temp_wl, b9_temp_bands, temp_px), key=lambda t: t[0]))
        temp_px = np.squeeze(np.dstack(temp_px)).T
        # Get the mean and stdev vectors and covariance matrix
        temp_avg = np.average(temp_px, axis=1)
        temp_cov = np.cov(temp_px)
        temp_stdev = np.std(temp_px, axis=1)
        header_str = "\\hline\n"
        header_str += "\\multicolumn{"+str(len(b9_temp_bands)+3)+"}{c}{" + \
                run_labels[i].replace("_"," ")+"} \\\\\n\\hline"
        rows = ""
        # Make a row for each band
        for j in range(len(b9_temp_bands)):
            row_str = f"{subgrid.info(b9_temp_bands[j])['ctr_wl']:.2f} & "
            row_str += f"{temp_avg[j]:.1f} & {temp_stdev[j]:.2f} & "
            row_str += " & ".join(map(
                lambda s:f"{s:.1f}",list(100*temp_cov[j,:]))) + " \\\\\n"
            rows += row_str
        if run_labels[i]=="uncertain":
            temp_uncertain = (header_str, col_str, rows)
        else:
            print(header_str)
            print(rows)
    print("\\end{tabular}")
    print("\\caption{"+run.replace("_", " ")+" brightness temp. statistics}")
    print("\\label{"+run+"_temp_stats}\n\\end{figure}")

    """ If there is an uncertain class, print a table for it """
    if temp_uncertain:
        print("\n\\clearpage\n")
        print("\\begin{figure}[h!]\n\\centering")
        print("\\begin{tabular}{C|C|C|" + "".join(
            ["C" for i in range(len(b9_ref_bands))]) + "}\n")
        print(ref_uncertain[1]+"\\hline"+ref_uncertain[2])
        print("\\end{tabular}")
        print("\\begin{tabular}{C|C|C|" + "".join(
            ["C" for i in range(len(b9_temp_bands))]) + "}\n")
        print(temp_uncertain[1]+"\\hline"+temp_uncertain[2])
        print("\\end{tabular}")
        print("\\caption{"+run.replace("_", " ") + \
                " Uncertain reflectance and brightness temp. statistics}")
        print("\\label{"+run+"_unc_stats}\n\\end{figure}")
    '''

    '''
    """ Get confusion matrices """
    # Order like (consumer, producer)
    #compare = ("km", "thresh")
    #compare = ("samples_thresh", "mlc_thresh")
    compare = ("samples_km", "mlc_km")
    L1, M1 = masks_dict[compare[0]]
    L2, M2 = masks_dict[compare[1]]
    #print("& " + " & ".join(L2))
    confusion = np.full((len(L1),len(L2)), 0)
    for j in range(len(L1)):
        for i in range(len(L2)):
            confusion[j,i] = np.count_nonzero(np.logical_and(M1[j],M2[i]))
    header_str = "& "+" & ".join(
            ["\\textnormal{"+l.replace("_"," ")+"}" for l in L2])
    header_str += " & \\textnormal{Cons. Acc.} \\\\\n\\hline"
    all_rows = ""
    for i in range(len(L1)):
        m = np.amax(confusion[i,:])
        s = np.sum(confusion[i,:])
        rowstr = "\\textnormal{"+L1[i].replace("_"," ")+"} & "+" & ".join(list(
            map(str, confusion[i,:])))
        rowstr += " & "+f"{m/s:.3f}"+ " \\\\\n"
        all_rows += rowstr
    all_rows += "\n\\hline\n\\textnormal{Prod. Acc.} & " + " & ".join([
        f"{np.amax(confusion[:,j])/np.sum(confusion[:,j]):.3f}"
        for j in range(len(L1))]) + "\\\\"
    print("\n\\begin{figure}[h!]\n\\centering")
    print("\\begin{tabular}{C|"+"".join(
        ["C" for i in range(len(L2))]) +"|C}")
    print(header_str)
    print(all_rows)
    print("\\end{tabular}")
    print("\\caption{"+compare[0].replace("_"," ") + "/" + \
            compare[1].replace("_"," ")+" confusion matrix}")
    print("\\label{confusion_"+"-".join(compare)+"}\n\\end{figure}")
    '''
