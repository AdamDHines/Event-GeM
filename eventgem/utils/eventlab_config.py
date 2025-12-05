import subprocess
import yaml, os
from pathlib import Path

def update_config(root, dataset, reference, query, time=50,
                  config_path='./eventgem/external/eventlab/config.yaml',
                  out_path=None):
    # Load the base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Edit the data_path entry to root
    # Make sure root is a full relative path so data isn't stored in eventlab folder
    full_root = Path(root).resolve()
    # convert full_root to a string¸
    path = full_root.as_posix()
    config['data_path'] = path

    # 2. Keep only one dataset with the sequences you care about
    config['datasets'] = [
        {
            'name': dataset,
            'sequences': [reference, query],
        }
    ]

    # 3. Update timewindows (optional, based on your argument)
    config['timewindows'] = [time]

    # 4. Make sure the frame_accumulator method is set to polarity
    config['frame_accumulator'] = 'polarity'

    # Where to save
    if out_path is None:
        out_path = config_path  # overwrite in-place

    with open(out_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

def eventlab_data():
    # Command to run Event-LAB
    command = ["pixi", "run", "-e", "default", "getdata", "config.yaml"]

    # Resolve Event-LAB path relative to this file
    this_dir = Path(__file__).resolve().parent
    eventlab_path = Path('./eventgem/external/eventlab')

    # Start from current environment, but drop pixi-related variables
    env = os.environ.copy()
    env.pop("PIXI_ENVIRONMENT", None)
    env.pop("PIXI_PROJECT_MANIFEST", None)

    # Now run pixi as if it was started fresh in the eventlab repo
    subprocess.run(command, cwd=eventlab_path, env=env, check=True)