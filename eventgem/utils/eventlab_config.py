import subprocess
import yaml, os, sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Event-LAB project root (where the "datasets" package and pixi.toml live)
EVENTLAB_ROOT = os.path.normpath(
    os.path.join(THIS_DIR, "..", "external", "eventlab")
)

if EVENTLAB_ROOT not in sys.path:
    sys.path.insert(0, EVENTLAB_ROOT)

from datasets.groundtruths import generate_ground_truth
from datasets.get_data import get_dataset


def update_config(root, dataset, reference, query, time=50,
                  config_path='./eventgem/external/eventlab/config.yaml',
                  dataset_base_path='./eventgem/external/eventlab/datasets',
                  out_path=None,
                  stream=False,
                  demo=False):

    # Load the base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the dataset config (used by generate_ground_truth)
    dataset_config_path = Path(dataset_base_path) / f"{dataset}.yaml"
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # 1. Edit the data_path entry to root (absolute path as POSIX string)
    full_root = Path(root).resolve()
    config['data_path'] = full_root.as_posix()

    # 2. Keep only one dataset with the sequences you care about
    if not demo:
        config['datasets'] = [
            {
                'name': dataset,
                'sequences': [reference, query],
            }
        ]
    else:
        # For the demo, we only want the query sequence to download, since we provide pre-generated descriptors for the reference
        config['datasets'] = [
            {
                'name': dataset,
                'sequences': [query],
            }
        ]

    # 3. Update timewindows
    config['timewindows'] = [time]

    # 4. Make sure the frame_accumulator method is set to polarity
    config['frame_accumulator'] = 'polarity'

    # Change the tolerance based on the dataset for ground truth
    if dataset == "brisbane_event" or dataset == "nsavp":
        config['ground_truth_tolerance'] = 70
    else:
        config['ground_truth_tolerance'] = 3 # for fast_slow

    # Where to save
    if out_path is None:
        out_path = config_path  # overwrite in-place

    with open(out_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Prior to running, if stream is set we can make the dataset folder to prevent Event-LAB from generating frames
    # This still lets Event-LAB download the raw data, but it won't take up space generating frames
    if stream:
        ref_path = f"{root}/{dataset}/{reference}/{reference}-frames-{time}"
        qry_path = f"{root}/{dataset}/{query}/{query}-frames-{time}"
        os.makedirs(ref_path, exist_ok=True)
        os.makedirs(qry_path, exist_ok=True)

    # -------- Run Event-LAB getdata via pixi (from EVENTLAB_ROOT) --------
    command = ["pixi", "run", "-e", "default", "getdata", "config.yaml"]

    env = os.environ.copy()
    env.pop("PIXI_ENVIRONMENT", None)
    env.pop("PIXI_PROJECT_MANIFEST", None)

    subprocess.run(command, cwd=EVENTLAB_ROOT, env=env, check=True)

    # -------- Generate ground truth (must also run from EVENTLAB_ROOT) --------
    prev_cwd = os.getcwd()
    try:
        os.chdir(EVENTLAB_ROOT)

        # These expect to find ./datasets/{dataset}.yaml etc.
        if not demo: # For the demo we only want the query dataset to download, pre-generated descriptors for referene will download
            ref_data = get_dataset(config, dataset, reference)
        query_data = get_dataset(config, dataset, query)

        if not demo:
            gps_available = dataset_config['sequences'][reference]['ground_truth']['available']

            generate_ground_truth(
                config,
                dataset_config,
                dataset,
                reference,
                query,
                ref_data,
                query_data,
                timewindow=time,
                gps_available=gps_available,
            )
    finally:
        os.chdir(prev_cwd)
