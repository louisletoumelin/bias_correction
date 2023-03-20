import matplotlib
import os
matplotlib.use('Agg')


def plot_1_1_subplot(df, key_obs="vw10m(m/s)", key_model="Wind", min_=-1, max_=30, s=1, figsize=(20,20)):
    import matplotlib.pyplot as plt
    # Get values
    obs = df[key_obs].values
    model = df[key_model].values

    # Get limits
    text_x = 0
    text_y = 21

    # Figure
    plt.figure(figsize=figsize)
    plt.scatter(obs, model, s=s)
    plt.plot(obs, obs, color='red')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)


def save_figure(name_figure, save_path, format_="png", svg=False):
    import matplotlib.pyplot as plt
    import uuid
    ax = plt.gca()
    fig = ax.get_figure()
    uuid_str = str(uuid.uuid4())[:4]
    fig.savefig(save_path + f"/{name_figure}_{uuid_str}.{format_}")
    if svg:
        fig.savefig(save_path + f"/{name_figure}_{uuid_str}.svg")


for station in test["name"].unique():
    print(station)
    plot_1_1_subplot(test[test["name"] == station], figsize=(10,10))
    if station in config["stations_test"]:
        save_path = os.path.join(os.getcwd(), "test/")
    elif station in config["stations_val"]:
        save_path = os.path.join(os.getcwd(), "val/")
    else:
        save_path = os.path.join(os.getcwd(), "train")
    save_figure(f"Wind_{station}", save_path)