#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%matplotlib inline

import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import math
import bisect
import secrets
import argparse
from scipy.stats import gamma
#import shutil
import decoy_consts
import mrl_decoy_plot

from scipy.stats import ks_2samp

uint64_t_MAX = 18446744073709551615
MIN_RCT_LENGTH = decoy_consts.CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE + 1
 
def GetParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dload-disable', default=False, action='store_true', help="Disable download?")
    parser.add_argument('-o', '--one-only', default=False, action='store_true', help="Plot only one")

    return parser

class GammaRVSPython():
    "Random variates"
    def __init__(self, shape=decoy_consts.GAMMA_SHAPE, scale=decoy_consts.GAMMA_SCALE):
        num_samples = 5000
        self.y = []
        for i in range(0, num_samples):
            self.y.append(gamma.rvs(a=shape, scale=scale))
        #print(self.y)


class GammaPDFPython():
    def __init__(self, shape=decoy_consts.GAMMA_SHAPE, scale=decoy_consts.GAMMA_SCALE):
        #define x-axis values
        self.x = np.linspace(4, 25, 100) 

        #calculate pdf of Gamma distribution for each x-value
        self.y = gamma.pdf(self.x, a=shape, scale=scale)

class GammaRVSMonero():
    def __init__(self):
        self.data = decoy_consts.load_data(decoy_consts.PATH_GAMMA_PDF)
        print(self.data)

def THROW_WALLET_EXCEPTION_IF(cond, context, msg):
    if cond:
        raise Exception(context + ": " + msg)

class GammaPickerPyhon():
    def __init__(self, rct_offsets, shape=decoy_consts.GAMMA_SHAPE, scale=decoy_consts.GAMMA_SCALE):
        #gamma = std::gamma_distribution<double>(shape, scale);
        self.rct_offsets = rct_offsets
        
        THROW_WALLET_EXCEPTION_IF(len(rct_offsets) <= decoy_consts.CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE, "error::wallet_internal_error", "Bad offset calculation")
        
        blocks_in_a_year = int(86400 * 365 / decoy_consts.DIFFICULTY_TARGET_V2)
        blocks_to_consider = min(len(rct_offsets), blocks_in_a_year)
        outputs_to_consider = rct_offsets[-1] - (rct_offsets[len(rct_offsets) - blocks_to_consider - 1] if blocks_to_consider < len(rct_offsets) else 0)
    
        begin = 0 # rct_offsets.data();
        # end = rct_offsets.data() + rct_offsets.size() - CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE;
        end = begin + len(rct_offsets) - decoy_consts.CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE
        #num_rct_outputs = *(end - 1);
        self.num_rct_outputs = rct_offsets[end - 1];
        THROW_WALLET_EXCEPTION_IF(self.num_rct_outputs == 0, "error::wallet_internal_error", "No rct outputs");
        self.average_output_time = decoy_consts.DIFFICULTY_TARGET_V2 * blocks_to_consider / float(outputs_to_consider); # // this assumes constant target over the whole rct range

    def rand_idx(self, maxVal):
        return secrets.randbelow(maxVal)

    def lower_bound(self, iterable, val_to_find):
        i = bisect.bisect_left(iterable, val_to_find)
        if i != len(iterable) and iterable[i] == val_to_find:
            return i
        return -1

    def pick_n_values_ratio(self, n):
        good = 0
        for i in range(0, n):
            val = self.pick()
            if val != uint64_t_MAX:
                good += 1
        return good / n

    def pick_n_values(self, n):
        ret = []
        for i in range(0, n):
            val = self.pick()
            if val != uint64_t_MAX:
                ret.append(val)
        return ret
            
        
    def pick(self):
        x = gamma.rvs(a=decoy_consts.GAMMA_SHAPE, scale=decoy_consts.GAMMA_SCALE)
          #double x = gamma(engine);
        x = math.exp(x);

        if (x > decoy_consts.DEFAULT_UNLOCK_TIME):
            x -= decoy_consts.DEFAULT_UNLOCK_TIME
        else:
            x = self.rand_idx(decoy_consts.RECENT_SPEND_WINDOW)

        output_index = int(x / self.average_output_time)
        if (output_index >= self.num_rct_outputs):
            return uint64_t_MAX; # bad pick
        output_index = self.num_rct_outputs - output_index; # TODO: Altered
        #if output_index == 0:
        #    output_index = 1

        #print("output_index", output_index)
        index = self.lower_bound(self.rct_offsets, output_index)
        if index < 0:
            # TODO: This was added!
            print("output_index of {} not found".format(output_index))
            return uint64_t_MAX # // bad pick
        THROW_WALLET_EXCEPTION_IF(index < 0, "error::wallet_internal_error", "output_index of {} not found".format(output_index))

        first_rct = 0 if index == 0 else self.rct_offsets[index - 1];
        n_rct = self.rct_offsets[index] - first_rct;
        if (n_rct == 0):
            return uint64_t_MAX # // bad pick
        #MTRACE("Picking 1/" << n_rct << " in block " << index);
        return first_rct + self.rand_idx(n_rct);

    """
      const uint64_t *it = std::lower_bound(begin, end, output_index);
      THROW_WALLET_EXCEPTION_IF(it == end, error::wallet_internal_error, "output_index not found");
      uint64_t index = std::distance(begin, it);

      const uint64_t first_rct = index == 0 ? 0 : rct_offsets[index - 1];
      const uint64_t n_rct = rct_offsets[index] - first_rct;
      if (n_rct == 0)
        return std::numeric_limits<uint64_t>::max(); // bad pick
      MTRACE("Picking 1/" << n_rct << " in block " << index);
      return first_rct + crypto::rand_idx(n_rct);
    """

class GammaPickerMonero():
    def __init__(self, rct_offsets):
        pass # TODO: This will only read the data file

def plot_picks(values):
    plt.plot(values)

    plt.show()

def picks(NUM_DRAWS=100, output_file=''):
    offsets_ratios = []

    mul = 1e5
    #mul = 1e3 # For testing
    while True:
        if mul <= 1: # TODO: Should be <= 1, but it crashes so far
            pass
            break
        num_hits = 0;
        start = 1 # At start == 0 there's a corner case to test
        rct_outputs = list(range(start, int(MIN_RCT_LENGTH * mul) + start))
        #print(rct_outputs)
        #print(len(rct_outputs))
        picker = GammaPickerPyhon(rct_outputs)
        ratio_good_picks = picker.pick_n_values_ratio(NUM_DRAWS)
        print(ratio_good_picks, len(rct_outputs))
        offsets_ratios.append((mul, ratio_good_picks))
        mul *= 0.85

    npa = np.array(offsets_ratios)
    if output_file:
        np.savetxt(output_file, npa, header='# multiplier_of_the_minimal_vector_length,ratio_good_picks')

    return npa

def picks_raw(NUM_DRAWS=100, output_file=''):
    offsets_ratios = []

    mul = 1e5
    #mul = 1e3 # For testing
    while True:
        if mul <= 1: # TODO: Should be <= 1, but it crashes so far
            pass
            break
        num_hits = 0;
        start = 1 # At start == 0 there's a corner case to test
        rct_outputs = list(range(start, int(MIN_RCT_LENGTH * mul) + start))
        #print(rct_outputs)
        #print(len(rct_outputs))
        picker = GammaPickerPyhon(rct_outputs)
        
        picks = picker.pick_n_values(NUM_DRAWS)
        print(len(rct_outputs))
        

        if output_file:
            fname = output_file + "_{}.csv".format(math.floor(round(mul)))
            np.savetxt(fname, picks)
            print("Saved to", fname)

        mul *= 0.85

    #npa = np.array(offsets_ratios)

    #return npa

def plot_data(gamRVSMo, gamRVSPy, gamPDFPy):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    bins = 50
    fig.suptitle("Gamma distributions' PDFs")
    ax1.hist(gamRVSMo.data, bins=bins)
    ax2.hist(gamRVSPy.y,    bins=bins)
    #ax2.plot(gamPDFPy.x, gamPDFPy.y)
    ax1.grid()
    ax2.grid()

    ax1.set_xlabel("Monero")
    ax2.set_xlabel("Python")
    ax1.set_ylabel("Occurrences")

    plt.show()

def plot_picker_py(ratios):
    mrl_decoy_plot.plot_cpp_distrib(ratios, "Python reimpl. gamma picker")

def ks(data1, data2):
    # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    print("Performing Kolmogorov-Smirnov test")
    print("Self-test: ", ks_2samp(data1[:, 1], data1[:, 1]))
    print("Final-test:", ks_2samp(data1[:, 1], data2[:, 1]))

def main():
    plot = True
    #plot = False
    parser = GetParser()
    args = parser.parse_args()
    start = 1 # At start == 0 there's a corner case to test
    #picks()
    picks_raw(100000, '/tmp/picks_raw_py_mul_length')

    #return
    #max_element = 20
    #data1 = data1[:-max_element]
    #data2 = data2[:-max_element]
    
    data_cpp = decoy_consts.load_data(decoy_consts.PATH_MUL_2_RATIO_GOOD)
    if plot:
        mrl_decoy_plot.plot_cpp_distrib(data_cpp, "Monero C++ gamma picker")
    
    rct_outputs = list(range(start, (decoy_consts.CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE + 1) * 1000 + start))

    if os.path.isfile(decoy_consts.PATH_MUL_2_RATIO_GOOD_PY):
        print("Loading", decoy_consts.PATH_MUL_2_RATIO_GOOD_PY)
        ratios = np.loadtxt(decoy_consts.PATH_MUL_2_RATIO_GOOD_PY)
    else:
        print("Generating", decoy_consts.PATH_MUL_2_RATIO_GOOD_PY)
        ratios = picks()
        np.savetxt(decoy_consts.PATH_MUL_2_RATIO_GOOD_PY, ratios)
    if plot:
        plot_picker_py(ratios)

    #return
    gamPDFPy = GammaPDFPython()
    gamRVSMo = GammaRVSMonero()
    gamRVSPy = GammaRVSPython()

    if plot:
        plot_data(gamRVSMo, gamRVSPy, gamPDFPy)
    ##plot_function(data)

    ks(data_cpp, ratios)

    MAX_NUM = 20
    NUM_DRAWS = 100000
    for i in range(0, MAX_NUM):
        print("Writing {} or {}".format(i, MAX_NUM))
        pass
        #ratios_new = picks(NUM_DRAWS, decoy_consts.PATH_MUL_2_RATIO_GOOD_PY_OUT + "_{}.csv".format(i))
        #ks(data_cpp, ratios_new)
    
if __name__ == "__main__":
    main()


