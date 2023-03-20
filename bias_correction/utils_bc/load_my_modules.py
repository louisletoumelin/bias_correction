import sys


def append_module_path(config, names=["downscale", "bias_correction"]):
    if "downscale" in names:
        sys.path.append(config["path_module_downscale"])
        print("\nLoaded downscale")
    if "bias_correction" in names:
        sys.path.append(config["path_root"]+"src/bias_correction/")
        print("\nLoaded bias_correction")
