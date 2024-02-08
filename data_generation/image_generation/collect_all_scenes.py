# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse, json, os
import numpy as np

"""
During rendering, each CLEVR scene file is dumped to disk as a separate JSON
file; this is convenient for distributing rendering across multiple machines.
This script collects all CLEVR scene files stored in a directory and combines
them into a single JSON file. This script also adds the version number, date,
and license to the output file.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)
parser.add_argument('--img_type', default='0')
parser.add_argument('--output_file', required=True)
parser.add_argument('--version', default='1.0')
parser.add_argument('--date', default='7/8/2017')
parser.add_argument('--license',
           default='Creative Commons Attribution (CC-BY 4.0')


def main(args):
  scenes = []
  split = []
  for task in ['t0', 't1', 't2']:
    for filename in os.listdir(f"{args.input_dir}/scenes/{task}/{args.img_type}"):
        if not filename.endswith('.json'):
            continue
        path = os.path.join(f"{args.input_dir}/scenes/{task}/{args.img_type}", filename)
        with open(path, 'r') as f:
            scene = json.load(f)
        scenes.append(scene)
        
        split.append(scene['split'])

  scenes.sort(key=lambda s: s['image_index'])

  for s in scenes:
    s['label'] = args.img_type
    print(s['image_filename'])

  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': np.unique(np.asarray(split)).tolist(),
      'license': args.license,
    },
    'scenes': scenes
  }

  with open(args.output_file, 'w') as f:
    json.dump(output, f)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

