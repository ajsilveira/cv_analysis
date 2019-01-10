import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyemma
from pyemma.thermo import tram
import logging
import yank

import os, psutil

def usage():
	process = psutil.Process(os.getpid())
	return process.memory_info()[0]/ (1024.0 ** 3)

logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
yank.utils.config_root_logger(verbose=True, log_file_path=None)
# Constants
T = 310
kT = 1.987E-3*T
axis_distance = 3.2 # nm
radius = 1.8 # nm
expansion_factor = 2.5

dr = axis_distance*(expansion_factor - 1.0)/2.0
rmax = axis_distance + dr
rmin = -dr

files = dict()
states = list()
for file in os.listdir('.'):
	if re.match(r'conf_[0-9]+\.txt', file):
		with open(file) as fp:
			for line in fp.read().split("\n")[:-1]:
				elements = line.split()
				state_string = " ".join([elements[1], elements[2], elements[3]])
				files[elements[0]] = state_string
				if state_string not in states:
					states.append(state_string)
logger.debug('memory after reading info of biased states')
logger.debug(usage())
with open('unbiased.txt') as fp:
	for line in fp.read().split("\n")[:-1]:
		elements = line.split()
		state_string = " ".join([elements[1], elements[2], '-1'])
		files[elements[0]] = state_string
		if state_string not in states:
			states.append(state_string)


logger.debug('memory after reading info of the unbiased state')
logger.debug(usage())
index_of_state = {}
for index, data in enumerate(states):
	index_of_state[data] = index

unbiased = len(states) - 1

del states

p_bins = 2000

dataframes = list()
rpmin = np.PINF
rpmax = np.NINF
state_values = list()
ttraj = []

for file_name, data in files.items():
	df = pd.read_csv(file_name, sep = '\s+' , names = ['r_p', 'r_o'])
	rpmin = min(rpmin, df['r_p'].min())
	rpmax = max(rpmax, df['r_p'].max())
	ttraj.append([index_of_state[data]]*len(df.index))
	if [int(value) for value in data.split()] not in state_values:
		state_values.append([int(value) for value in data.split()])
	del df


logger.debug('memory after ttraj')
logger.debug(usage())
ncenters = np.max(np.asarray(state_values), axis=0)[2] + 1
nstates = len(state_values)
for i, state in enumerate(state_values):
	if (state[2] >= 0):
		state_values[i][0] = state_values[i][0]/kT
		state_values[i][1] = state_values[i][1]/kT
		state_values[i][2] = (rmax - rmin)*state[2]/(ncenters - 1) + rmin
	else:
		state_values[i][2] = 0


dtraj = []
bias = []
factor = p_bins/(rpmax - rpmin)

for file_name, data in files.items():
	df = pd.read_csv(file_name, sep = '\s+' , names = ['r_p', 'r_o'])
	logger.debug(file_name)
	dtraj.append(np.floor(factor*(df['r_p'] - rpmin)).astype(int).tolist())
	nconfs = len(df.index)
	local_bias = np.ndarray(shape=(nconfs, nstates))
	for i, state in enumerate(state_values):
		local_bias[:,i] = (state[0]/2.0)*(df['r_p'].values-state[2])**2 + (state[1]/2.0)*(df['r_o'].values**2)
	del df
	bias.append(local_bias.tolist())
	logger.debug('memory after appending bias')
	logger.debug(usage())
	del local_bias


#tram_obj = tram(ttraj, dtraj, bias, unbiased_state)
#f = open('free_energies.txt','w')
#f.write(tram_obj.free_energies)


# for i, b in enumerate(bias):
# 	print("\n\nlist {}".format(i))
# 	for j, c in enumerate(b):
# 		print("size of sublist[{}] = {}".format(j, len(c)))
