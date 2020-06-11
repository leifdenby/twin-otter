"""
For finding and analysing boundary layer legs
"""
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
import scipy.ndimage
from scipy import stats
from tqdm import tqdm

from .. import load_flight


def make_timeseries_plot(ds, variables, include_velocity_arrows=True):
    """
    Call with list of variables, if tuple is provided the variables will be
    printed on twin axes

    e.g.
    >> variables = ["ALT_OXTS", ("TDEW_BUCK", "H2O_LICOR"), "HUM_ROSE", ]
    >> make_timeseries_plot(ds, variables)
    """
    fig, axes = plt.subplots(
        nrows=len(variables), figsize=(16, 3 * len(variables)), sharex=True
    )

    if include_velocity_arrows:
        s = slice(None, None, 50)
        u = ds.VELE_OXTS[s]
        v = ds.VELN_OXTS[s]
        x_ = ds.time[s].values
        y_ = ds.ALT_OXTS.max().values * 1.1 + np.zeros_like(u)

    axes[0].quiver(x_, y_, u, v, scale=5.0, scale_units="dots", width=1.0e-3)

    for n, v in enumerate(variables):
        if type(v) in [list, tuple]:
            ax1 = axes[n]
            ax2 = axes[n].twinx()
            (l1,) = ds[v[0]].plot(ax=ax1, marker=".", label=v[0])
            (l2,) = ds[v[1]].plot(ax=ax2, marker=".", label=v[1], color="orange")
            lines = [l1, l2]
            ax1.legend(lines, [l.get_label() for l in lines])
        else:
            ds[v].plot(ax=axes[n], marker=".")
    sns.despine(fig)


def find_bl_legs(
    da_z, z_max=600.0, dt_min=datetime.timedelta(minutes=10), return_labels=False
):
    """
    Find all boundary layer legs (height below z_max) with a minimum duration of `dt_min`
    """
    ds_ = da_z.to_dataset()
    is_steady = np.abs(da_z.differentiate("time")) < 3.0e-9
    ds_["is_bl"] = np.logical_and(da_z < z_max, is_steady)
    ds_.is_bl.attrs = dict(units="1", long_name="is boundary leg")

    ds_["label"] = da_z.dims, scipy.ndimage.label(ds_.is_bl)[0]
    ds_.label.attrs = dict(units="1", long_name="BL leg label")

    legs = []
    for i_label in np.unique(ds_.label)[1:]:
        ds_leg = ds_.where(ds_.label == i_label, drop=True)
        dt_xr = ds_leg.time.max() - ds_leg.time.min()
        ds_seconds = dt_xr.dt.seconds + 24 * 60 * 60 * dt_xr.dt.days
        if ds_seconds > dt_min.total_seconds():
            legs.append(ds_leg)
            ds_leg["label"] = len(legs)

    if return_labels:
        ds_concat = xr.concat(legs, dim="time")
        ds_concat = ds_concat.drop(da_z.name)
        return ds_concat
    else:
        leg_times = []
        for ds_leg in legs:
            leg_times.append((ds_leg.time.min(), ds_leg.time.max()))
        return leg_times


def main():
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("flight_data_path")
    args = argparser.parse_args()

    generate_flight_plots(args.flight_data_path)


def _rename_time(ds_):
    if "Time" in ds_.coords:
        da_time = ds_.Time
        da_time.name = "time"
        ds_["time"] = da_time
        ds_ = ds_.swap_dims(dict(Time="time"))
        ds_ = ds_.drop("Time")
    return ds_


def generate_flight_plots(flight_data_path):
    ds = load_flight(flight_data_path, debug=True)
    ds = _rename_time(ds)

    ds_bl_legs = find_bl_legs(ds.ALT_OXTS, return_labels=True)
    ds_merged = xr.merge([ds, ds_bl_legs])

    import seaborn as sns

    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2

    x = "TDEW_BUCK"
    y = "H2O_LICOR"

    for v in [x, y]:
        if v not in ds:
            raise Exception(f"No `{v}` data")

    p_figures = Path(flight_data_path) / "figures" / "boundary_layer"
    p_figures.mkdir(exist_ok=True)

    bl_legs = find_bl_legs(da_z=ds.ALT_OXTS)
    for n, (t_min, t_max) in enumerate(tqdm(bl_legs)):
        ds_leg = ds_merged.sel(time=slice(t_min, t_max))
        plt.figure()
        sns.jointplot(x, y, data=ds_leg, kind="reg", stat_func=r2)
        plt.gcf().suptitle(f"flight {ds.flight_number}, boundary layer leg {n}\n"
                           f"{t_min.values} to {t_max.values}")
        plt.savefig(str(p_figures/f"joint_dist_{n}.png"), bbox_inches="tight")

        plt.figure()
        variables = ["ALT_OXTS", ("TDEW_BUCK", "H2O_LICOR"), "HUM_ROSE", "W_OXTS"]
        make_timeseries_plot(ds_leg, variables)
        st = plt.suptitle(f"Flight {ds.flight_number} boundary layer leg {n}")
        plt.savefig(str(p_figures/f"timeseries_{n}.png"), bbox_inches="tight",
                     bbox_extra_artists=[st],)


if __name__ == "__main__":
    main()
