"""Detect rurociag.
Usage:
  read_mask.py <img> <out_dir>
  read_mask.py -h | --help


Options:
  -h --help     Help
  <img>         Image path
  <out_dir>     Output dir path
"""

import os

from docopt import docopt
from PIL import Image

from generic_segmenter import *


if __name__ == '__main__':
    arguments = docopt(__doc__)
    img = arguments['<img>']

    res_name = img.split(r'/')[-1].split('.')[0]+'_out.png'

    model = YoloV7_segmenter(r'ml_models/droniada_rurociag.pt')

    img_out = model.segment(Image.open(img))
    
    out_path = os.path.join(arguments['<out_dir>'], res_name)

    img_out.save(out_path, 'PNG')