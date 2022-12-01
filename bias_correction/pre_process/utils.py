def append_module_path(config):
    import sys
    sys.path.append(config["path_module_downscale"])
    #sys.path.append(config["path_root"]+"src/bias_correction/")
    print("Module path added")
