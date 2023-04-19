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
thresh_pkl = data_dir.joinpath(f"subgrid_masks.pkl")
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

def get_raw_images(subgrid, fig_dir):
    """ Get a grayscale and HSV scalar raster for each band """
    print(TFmt.YELLOW("Generating grayscale rasters for all bands"))
    for b in l1b_bands:
        gp.generate_raw_image(enh.norm_to_uint(subgrid.data(b),256,np.uint8),
                              fig_dir.joinpath(Path(f"scalar/gs_{b:02}.png")))
        gp.generate_raw_image(
                enh.norm_to_uint(1-gt.scal_to_rgb(subgrid.data(b)),
                                 256, np.uint8),
                fig_dir.joinpath(Path(f"scalar/hsv_{b:02}.png")))
        if subgrid.info(b)["is_reflective"]:
            gp.generate_raw_image(
                    enh.norm_to_uint(1-gt.scal_to_rgb(
                        subgrid.raw_reflectance(b)),256,np.uint8),
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
        #mask_labels, all_masks = tuple(zip(*masks_dict.items()))
        #not_water_mask = np.any(np.dstack(all_masks),axis=2)
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
        mask_labels, all_masks = tuple(zip(*masks_dict.items()))
        not_water_mask = np.any(np.dstack(all_masks),axis=2)
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

    subgrid = restore_subgrid(subgrid_pkl)
    #get_raw_images(subgrid, fig_dir)
    #get_rgbs(subgrid, fig_dir, choose=True, gamma_scale=4)
    #get_surface_masks(subgrid, fig_dir, thresh_pkl)
    #get_sg2_surface_masks(subgrid, fig_dir, thresh_pkl)
    #do_threshold_spectral(subgrid, thresh_pkl)

    '''
    """ Do spectral analysis on my custom RGB """
    subgrid.add_recipe("Band 29, Norm",
                       Recipe((29,),lambda a: enh.linear_gamma_stretch(a)))
    subgrid.add_recipe("NDVI Norm",
                       Recipe(("ndvi",),lambda a: enh.linear_gamma_stretch(a)))
    subgrid.add_recipe("Inv. Band 31, Norm",
                       Recipe((31,),lambda a: 1-enh.linear_gamma_stretch(a)))
    subgrid.histogram_analysis(
            labels=["Band 29, Norm","NDVI Norm","Inv. Band 31, Norm"],
            nbins=1024,
            plot_spec={"colors":[[1,0,0],[0,1,0],[0,0,1]]},
            show=True)
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
    km_bands=(3,4,1,2,7,19,29,31,32,33) # Batch 8 w/ NIR abs.
    #km_bands=(1,2,3,4,5,19,20,26,28,29,31) # Batch 9; batch 2 w/20 not 21
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
    new_km[np.where(km==2)] = 2 # mid clouds
    new_km[np.where(km==6)] = 2 # mid clouds
    new_km[np.where(km==7)] = 2 # mid clouds
    new_km[np.where(km==9)] = 3 # cirrus
    new_km[np.where(km==3)] = 3 # cirrus
    new_km[np.where(km==4)] = 4 # arid
    new_km[np.where(km==5)] = 4 # arid

    km_dict["merged"] = (new_km, km_bands)
    #pkl.dump(km_dict, kmeans_pkl.open("wb"))
    # Subjective class ID order of batch9c10 classification
    km_order = ["vegetation", "water", "water_cloud", "ice_cloud", "arid"]
    km_colors = { colors[l] for l in km_order }
    do_kmeans(
            subgrid,
            km_bands=km_bands,
            # Be very careful with label keys! Don't overwrites something.
            #batch_label="merged",
            km_class_count=5,
            km_pkl=kmeans_pkl,
            fig_dir=fig_dir,
            title=f"Merged Batch 9 Classes",
            get_new=False,
            plot_classes=True,
            colors=km_order,
            )
    '''

    '''
    km_masks = [np.full_like(km, False, dtype=bool) for i in range(8)]
    for j in range(km.shape[0]):
        for i in range(km.shape[1]):
            km_masks[km[j,i]][j,i] = True
    '''

    #'''
    """ Do maximum-likelihood classification """
    samples = 400
    thresh = .9
    #mlc_bands = (1,2,3,4,5,19,20,26,28,29,31) # Batch 9
    mlc_bands = None
    mlc_batch = "all-thresh"
    #mlc_batch = "b9-thresh"
    use_km = False
    #mlc_batch = "all-km"
    #mlc_batch = "b9-km"
    title = f"Maximum-likelihood Classification Results ({mlc_batch})"

    if use_km:
        """ Get samples from K-means results """
        km_order = ["vegetation", "water", "water_cloud", "ice_cloud", "arid"]
        km_colors = [ colors[l] for l in km_order ]
        if thresh:
            km_colors.append(colors["uncertain"])
        km_ints_merged, km_bands = pkl.load(kmeans_pkl.open("rb"))["merged"]
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
    #'''
    exit(0)

    """ Calculate the confusion matrices """
    # Returns the number of elements in common between masks
    confusion = lambda m1,m2: np.count_nonzero(np.logical_and(m1,m2))
    km_ints, km_bands = pkl.load(kmeans_pkl.open("rb"))["merged"]
    km_masks = MOD021KM.ints_to_masks(km_ints)
    thresh_labels, thresh_masks = zip(*[
        (c,MOD021KM.mask_to_idx(M, samples))
        for c,M in pkl.load(thresh_pkl.open("rb")).items()])
    mlc_run = "b9-km"
    mlc_ints, mlc_labels = pkl.load(mlc_pkl.open("rb"))[mlc_run]
    samples_pixels, samples_labels = pkl.load(samples_pkl.open("rb"))[mlc_run]
    MOD021KM.idx_to_mask(
            [list(zip(*XY)) for XY in samples_pixels],subgrid.shape)

    for c in range(len(class_covs)):
        for j in class_covs[c].shape[0]:
            row_str = ""
            for i in class_covs[c].shape[1]:
                row_str+=f" {class_covs[c][j,i]:.3f} "
            print(row_str)
