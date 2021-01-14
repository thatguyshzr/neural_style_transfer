import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

from PIL import Image
import argparse

from visualize_image import *
from optimize_loop import *

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

parser = argparse.ArgumentParser()

parser.add_argument('--input', '--in', help="input image", required=True)
parser.add_argument('--style', '--s', help="style image", required=True)
parser.add_argument('--iter', '--it', help="number of iterations", required=False,
                    type= int, default=5)

args = parser.parse_args()

best_image, best_loss = run_style_transfer(args.input, args.style, args.iter)

show_results(best_image, args.input, args.style)
