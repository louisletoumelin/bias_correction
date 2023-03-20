import pandas as pd
import xarray as xr

from bias_correction.pre_process.config_preprocess import config
from bias_correction.pre_process.utils import append_module_path
append_module_path(config)
from bias_correction.pre_process.stations import Stations
from bias_correction.pre_process.nwp import Nwp
from bias_correction.pre_process.time_series import TimeSeries
from bias_correction.pre_process.topo import DictTopo

# Preprocess wiss Data: see preprocess_swiss_data.ipynb
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

# Stations
if config["pre_process_stations"]:
    stations = pd.read_csv(config["path_station"] + "stations_alps.csv")
    dem = xr.open_dataset(config["path_dem"] + "DEM_ALPES_L93_30m.nc")
    nwp_france = xr.open_dataset(config["path_nwp_alp"] + "AROME_alp_2017_10.nc")
    nwp_swiss = xr.open_dataset(config["path_nwp_swiss"] + "AROME_switzerland_2017_11.nc")

    s = Stations(stations=stations, nwp_france=nwp_france, nwp_swiss=nwp_swiss, dem=dem, config=config)
    s.convert_lat_lon_to_L93()
    s.update_stations_with_KNN_from_MNT_using_cKDTree()
    s.update_stations_with_KNN_from_NWP(interpolated=False)
    s.update_stations_with_KNN_of_NWP_in_MNT_using_cKDTree(interpolated=False)
    s.update_station_with_topo_characteristics()
    s.interpolate_nwp()
    s.update_stations_with_KNN_from_NWP(interpolated=True)
    s.update_stations_with_KNN_of_NWP_in_MNT_using_cKDTree(interpolated=True)
    s.change_dtype_stations()
    s.save_to_pickle()

    del s

# nwp
if config["pre_process_nwp"]:
    n = Nwp(config)
    n.check_all_lon_and_lat_are_the_same_in_nwp() # remote
    n.save_L93_npy() # local, do it once
    n.print_send_L93_npy_to_labia()# Send the X_L93.pny to the labia, do it once
    n.add_L93_to_all_nwp_files() # remote
    n.downcast_to_float32()
    #todo se renseigner sur les Z0
    #n.add_Z0_to_all_nwp_files() # remote, impossible because we don't have Z0 for Switzerland

    del n

# time series
if config["pre_process_time_series"]:
    stations = pd.read_pickle(config["path_station_pre_processed"] + "stations_alps.csv")
    time_series = pd.read_csv(config["path_time_series"] + "time_series_alps.csv")
    t = TimeSeries(time_series=time_series, stations=stations, config=config)
    #time_series = apply_qc(time_series)
    # We suppose all stations in Switzerland measure wind speed at 10 m a.g.l.
    t.keep_minimal_variables()
    t.add_AROME_variables() # Remote because of memory problems
    t.compute_u_and_v() # Remote
    # todo modify change_dtype_time_series with new variables after qc
    t.change_dtype_time_series()
    t.save_to_pickle()

    del t

# topos
if config["pre_process_topos"]:
    d = DictTopo(stations=stations, dem=dem, config=config)
    d.store_topo_in_dict()

    del d

