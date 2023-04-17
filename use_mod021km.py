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

def get_raw_images(subgrid, fig_dir):
    """ Get a grayscale and HSV scalar raster for each band """
    print(TFmt.YELLOW("Generating grayscale rasters for all bands"))
    for b in l1b_bands:
        gp.generate_raw_image(enh.norm_to_uint(subgrid.data(b), 256, np.uint8),
                              fig_dir.joinpath(Path(f"scalar/raw_{b:02}.png")))
        gp.generate_raw_image(
                enh.norm_to_uint(1-gt.scal_to_rgb(subgrid.data(b)),
                                 256, np.uint8),
                fig_dir.joinpath(Path(f"hsv/hsv_{b:02}.png")))

def get_surface_masks(subgrid, fig_dir, masks_pkl):
    """ """
    def _update_masks_pkl(mask_key, mask):
        # Load already-selected masks back from the pkl
        stored_masks = pkl.load(masks_pkl.open("rb"))
        stored_masks[mask_key] = mask
        # Load the new masks back into the pkl
        pkl.dump(stored_masks, masks_pkl.open("wb"))

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
        _update_masks_pkl(mask_key, inv_mask)
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
        _update_masks_pkl(mask_key, inv_mask)
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
        #_update_masks_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))
        old_igneous_mask = np.logical_and(inv_mask, oig_mask)
        new_igneous_mask = np.logical_and(inv_mask, np.logical_not(oig_mask))
        _update_masks_pkl("old_igneous", old_igneous_mask)
        _update_masks_pkl("new_igneous", new_igneous_mask)

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
        _update_masks_pkl(mask_key, inv_mask)
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
        masks_dict = pkl.load(masks_pkl.open("rb"))
        mask_labels, all_masks = tuple(zip(*masks_dict.items()))
        not_water_mask = np.any(np.dstack(all_masks),axis=2)
        not_water_mask = np.logical_or(not_water_mask, mask)
        water_mask = np.logical_not(not_water_mask)
        rgb[np.where(np.dstack([not_water_mask for i in range(3)]))] = 0

        # Invert the mask so that water are True
        inv_mask = np.logical_not(not_water_mask)
        _update_masks_pkl(mask_key, inv_mask)
        gp.generate_raw_image(
                rgb, fig_dir.joinpath(f"rgbs/mask_{mask_key}.png"))

def do_threshold_spectral(subgrid, masks_pkl):
    """
    Generate spectral distributions for pixels that belong to each mask.
    """
    mask_labels, masks = tuple(zip(*pkl.load(masks_pkl.open("rb")).items()))
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

if __name__=="__main__":
    """ Settings """
    img_template = "%Y%m%d_%H%M_{label}_{sat}.png"
    buf_dir = Path("./buffer")
    fig_dir = Path("./figures")
    data_dir = Path("./data")
    target_latlon = (19.588, -155.51)
    target_time = dt(year=2017, month=6, day=13, hour=20, minute=54)
    satellite = "terra"
    region_width, region_height = 640, 512
    debug = True
    hdf_path = data_dir.joinpath(f"subgrid.hdf")
    subgrid_pkl = data_dir.joinpath(f"subgrid.pkl")
    kmeans_pkl = data_dir.joinpath(f"subgrid_kmeans.pkl")
    masks_pkl = data_dir.joinpath(f"subgrid_masks.pkl")
    subgrid = restore_subgrid(subgrid_pkl)
    #get_raw_images(subgrid, fig_dir)
    #get_surface_masks(subgrid, fig_dir, masks_pkl)
    #do_threshold_spectral(subgrid, masks_pkl)

    km_class_count = 6
    get_new = True
    km_arr = pkl.load(kmeans_pkl.open("rb"))
    kmeans_bands = (1,2,3,4,5,19,21,26,28,29,31)
    batch_label = "batch2"
    if get_new:
        new_batch = subgrid.get_kmeans(
                labels=kmeans_bands, class_count=km_class_count,
                return_as_ints=True, debug=True)
        km_arr[batch_label] = (new_batch, kmeans_bands)
        pkl.dump(km_arr, kmeans_pkl.open("wb"))

    gp.plot_classes(
            class_array=km_arr[batch_label][0],
            class_labels=[f"KM{i+1}" for i in range(km_class_count)],
            colors=gt.unique_colors(km_class_count),
            plot_spec={
                "title":f"All-band K-means results ({km_class_count} classes)"
                },
            fig_path=fig_dir.joinpath(
                #f"class/kmeans_all_{km_class_count}c.png"),
                f"class/kmeans_{batch_label}_{km_class_count}c.png"),
            show=False,
            )
