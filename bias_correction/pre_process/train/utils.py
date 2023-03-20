import matplotlib

from contextlib import contextmanager
import os

#matplotlib.use('Agg')


def plot_1_1_subplot(df, key_obs="vw10m(m/s)", key_model="Wind", min_=-1, max_=30, s=1, figsize=(20,20)):
    import matplotlib.pyplot as plt

    # Get values
    obs = df[key_obs].values
    model = df[key_model].values

    # Figure
    plt.figure(figsize=figsize)
    plt.scatter(obs, model, s=s)
    plt.plot(obs, obs, color='black')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)


def save_figure(name_figure, save_path, format_="png", svg=False, fig=None):
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import uuid
    ax = plt.gca()
    if fig is None:
        fig = ax.get_figure()
    uuid_str = str(uuid.uuid4())[:4]
    fig.savefig(save_path + f"/{name_figure}_{uuid_str}.{format_}")
    if svg:
        fig.savefig(save_path + f"/{name_figure}_{uuid_str}.svg")


@contextmanager
def no_raise_on_key_error():
    try:
        yield
    except KeyError:
        pass


class FolderShouldNotExistError(Exception):
    pass


def create_folder_if_doesnt_exist(path: str,
                                  _raise: bool = True,
                                  verbose: bool = False
                                  ) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    elif _raise:
        raise FolderShouldNotExistError(path)
    else:
        if verbose:
            print(f"{path} already exists")
        pass

