import tensorflow as tf

import os

from bias_correction.train.eval import Interpretability


class FeatureImportanceCallback(tf.keras.callbacks.Callback):

    def __init__(self, data_loader, cm, exp, mode):
        self.it = Interpretability(data_loader, cm, exp)
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        df_rmse, df_ae, _, _ = self.it.compute_feature_importance(self.mode)
        name_figure = str(int(epoch))
        save_path = os.path.join(self.it.exp.path_to_feature_importance, name_figure, "figure_tmp.tmp")
        subfolder_in_filename = self.it.check_if_subfolder_in_filename(save_path)
        if subfolder_in_filename:
            new_path, _ = self.it.create_subfolder_if_necessary(name_figure, save_path)
        df_rmse.to_csv(os.path.join(save_path, "df_rmse.csv"))
        df_ae.to_csv(os.path.join(save_path, "df_ae.csv"))
        print("Feature importance computed and saved")
