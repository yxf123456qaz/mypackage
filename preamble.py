
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle
import itertools
import math
import sys, os
from pandas import read_csv, Series
from numpy import mat,cov, mean, hstack, multiply,sqrt,diag,squeeze, ones, array, vstack, kron, zeros, eye,tile, reshape, squeeze, eye, asmatrix
from numpy.linalg import inv
from datetime import datetime as dt
from scipy.stats import chi2
from scipy import signal
from statsmodels.sandbox.regression import gmm
from IPython.display import display, Math, Latex
from scipy.linalg import kron,pinv
from scipy.optimize import fmin_bfgs,fmin
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 1000)
np.set_printoptions(edgeitems=3, threshold=100, suppress=True, linewidth=1000)