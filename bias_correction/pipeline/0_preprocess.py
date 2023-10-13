import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.append("/home/mrmn/letoumelinl/bias_correction/src/")
print(sys.path)
import logging

from bias_correction.config.config_preprocess import config
from bias_correction.utils_bc.load_my_modules import append_module_path
append_module_path(config, names=["downscale"])
from bias_correction.pre_process.stations import Stations
from bias_correction.pre_process.nwp import Nwp
from bias_correction.pre_process.time_series import TimeSeries
from bias_correction.pre_process.topo import DictTopo

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Preprocess Swiss Data: see preprocess_swiss_data.ipynb
#idaweb_format_to_single_station_csv()
#drop_hns000hs() # Keep automatic values for HTN only
#rename_columns
#date_correct_format
#raise_if_dupicates_in_time_series
#resample_1h_keep_first()
#keep_only_station_where_speed_is_measured
#remove_negative_htn
#if_abs_difference_between_htn_and_rolling_mean_superior_50cm_write_nan
#find_peaks_threshold30_width_0_5
#interpolate_missing_values
#merge_clean_csv_and_save

NAME = "flowcapt"

# Stations
if config["pre_process_stations"]:
    stations = pd.read_csv(config["path_station"] + "stations_isaw.csv")
    dem = xr.open_dataset(config["path_dem"] + "DEM_ALPES_L93_30m.nc")
    dem_pyr_corse = xr.open_dataset(config["path_dem"] + "DEM_FRANCE_L93_30m_bilinear.nc")
    nwp_france = xr.open_dataset(config["path_nwp_alp"] + "AROME_alp_2017_10.nc")
    nwp_swiss = xr.open_dataset(config["path_nwp_swiss"] + "AROME_switzerland_2017_11.nc")
    #nwp_pyr = xr.open_dataset(config["path_nwp_pyr"] + "AROME_pyr_2019_11.nc")
    #nwp_corse = xr.open_dataset(config["path_nwp_corse"] + "AROME_corse_2019_11.nc")

    s = Stations(stations=stations,
                 nwp_france=nwp_france,
                 nwp_swiss=nwp_swiss,
                 dem=dem,
                 dem_pyr_corse=dem_pyr_corse,
                 config=config)

    s.convert_lat_lon_to_l93()  # Robust to new stations, tested=no
    print(stations[["X", "Y"]].describe())
    s.update_stations_with_knn_from_mnt_using_ckdtree()  # Robust to new stations, tested=no
    s.update_stations_with_knn_from_nwp(interpolated=False)  # Robust to new stations, tested=no
    s.update_stations_with_knn_of_nwp_in_mnt_using_ckdtree(interpolated=False)  # Robust to new stations, tested=no
    #s.update_station_with_topo_characteristics()  # Robust to new stations, tested=no
    s.interpolate_nwp()  # Robust to new stations, tested=no
    s.update_stations_with_knn_from_nwp(interpolated=True)  # Robust to new stations, tested=no
    s.update_stations_with_knn_of_nwp_in_mnt_using_ckdtree(interpolated=True)  # Robust to new stations, tested=no
    #s.change_dtype_stations()  # Robust to new stations, tested=no
    s.save_to_pickle(name=NAME)  # Robust to new stations, tested=no

    del s


# nwp
if config["pre_process_nwp"]:
    n = Nwp(config)
    n.check_all_lon_and_lat_are_the_same_in_nwp()         # Remote,                                                Done
    nwp_pyr = xr.open_dataset(config["path_nwp_pyr"] + "AROME_pyr_2019_11.nc")
    nwp_corse = xr.open_dataset(config["path_nwp_corse"] + "AROME_corse_2019_11.nc")
    n.compute_l93(nwp_pyr, "pyr")                        # Local, do it once,                                      Done
    n.compute_l93(nwp_corse, "corse")                    # Local, do it once,                                      Done
    n.save_L93_npy()                                      # Local, do it once,                                     Done
    n.print_send_L93_npy_to_labia()                       # Local to labia, do it once                             Done
    n.add_L93_to_all_nwp_files()                          # Remote, do it once                                     Done
    n.downcast_to_float32()                               # do it once                                             Done

    # todo se renseigner sur les Z0
    n.add_Z0_to_all_nwp_files()                           # Remote, impossible because we don't have Z0 for Switzerland

    del n

# time series
if config["pre_process_time_series"]:

    stations = pd.read_pickle(config["path_stations_pre_processed"] + f"stations_bc{NAME}.pkl")
    time_series = pd.read_pickle(config["path_time_series_pre_processed"] + "time_series_with_clb.pkl")

    for interpolated in [False]:

        logger.warning(interpolated)
        t = TimeSeries(time_series=time_series, stations=stations, interpolated=interpolated, config=config)

        # We suppose all stations in Switzerland measure wind speed at 10 m a.g.l.
        #time_series = apply_qc(time_series)
        logger.warning("\nTimeSeries instance created")

        logger.warning("\nbegin select_date")
        t.select_date()                                                                              # R, no tested
        logger.warning("select_date done")

        logger.warning("\nbegin keep_minimal_variables")
        t.keep_minimal_variables()                            # Remote                               # R, no tested
        logger.warning("keep_minimal_variables done")

        logger.warning("\nbegin add_arome_variables")
        t.add_arome_variables()                               # Remote because of memory problems    # R, no tested
        logger.warning("add_arome_variables done")

        logger.warning("\nbegin compute_u_and_v")
        t.compute_u_and_v()                                   # Remote                               # R, no tested
        logger.warning("compute_u_and_v done")

        logger.warning("\nbegin change_dtype_time_series")
        try:
            t.change_dtype_time_series()
        except:
            logger.warning("\n\nchange_dtype_time_series raised an error")    # Remote               # R, no tested
        logger.warning("change_dtype_time_series done")

        logger.warning("\nbegin save_to_pickle")
        t.save_to_pickle()                                    # Remote                               # R, no tested
        logger.warning("save_to_pickle done")

    del t

# topos
if config["pre_process_topos"]:

    stations = pd.read_pickle(config["path_stations_pre_processed"] + f"stations_bc{NAME}.pkl")
    dem = xr.open_dataset(config["path_dem"] + "DEM_ALPES_L93_30m.nc")
    dem_pyr_corse = xr.open_dataset(config["path_dem"] + "DEM_FRANCE_L93_30m_bilinear.nc")

    d = DictTopo(stations=stations,
                 dem=dem,
                 dem_pyr_corse=dem_pyr_corse,
                 config=config)

    d.store_topo_in_dict(name=NAME)  # R, no tested

    del d

