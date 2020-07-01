import copy
from pathlib import Path
import pickle

import fire

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database


def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, save_path=None):
    if save_path is None:
        save_path = root_path
    nu_ds.create_nuscenes_infos(root_path, save_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(save_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
        db_path=Path(root_path) / f"gt_database_{nsweeps}sweeps_withvelo_centerpoint",
        dbinfo_path=Path(save_path) / f"dbinfos_train_{nsweeps}sweeps_withvelo.pkl",
        nsweeps=nsweeps,
    )


if __name__ == "__main__":
    fire.Fire()
