import os, sys, pickle, scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
import matplotlib.patches as mpatches

from glob import glob
from numpy import mean
from Anlz.utils_anlz import cal_theta, euclidean_distance
from pathlib import Path

class RSA_Anlz:
    def __init__(self, subj_name, net_name, trained, root_dir):
        self.subj_name = subj_name
        self.model_name = net_name
        self.trained = 'Trained' if trained else 'Untrained'

        self.root_dir = root_dir
        self.save_dir = Path(self.root_dir, 'Result', 'RSA')

    def extract():
        pass
