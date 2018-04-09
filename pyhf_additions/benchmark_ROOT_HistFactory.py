# This needs to be run _inside_ of pyhf/validation/multibin_histfactory/
import ROOT
from ROOT import gROOT

import os
import shutil
from subprocess import call
import json
import numpy as np
import time


def generate_source(n_bins):
    binning = [n_bins, -0.5, n_bins + 0.5]
    data = [120.0] * n_bins
    bkg = [100.0] * n_bins
    bkgerr = [10.0] * n_bins
    sig = [30.0] * n_bins
    source = {
        'binning': binning,
        'bindata': {
            'data': data,
            'bkg': bkg,
            'bkgerr': bkgerr,
            'sig': sig
        }
    }
    return source


def generate_data(source, file_name='data/data.root'):
    source_data = source

    binning = source_data['binning']
    bindata = source_data['bindata']

    f = ROOT.TFile(file_name, 'RECREATE')
    data = ROOT.TH1F('data', 'data', *binning)
    for i, v in enumerate(bindata['data']):
        data.SetBinContent(i + 1, v)
    data.Sumw2()

    bkg = ROOT.TH1F('bkg', 'bkg', *binning)
    for i, v in enumerate(bindata['bkg']):
        bkg.SetBinContent(i + 1, v)
    bkg.Sumw2()

    if 'bkgerr' in bindata:
        bkgerr = ROOT.TH1F('bkgerr', 'bkgerr', *binning)

        # shapesys must be as multiplicative factor
        for i, v in enumerate(bindata['bkgerr']):
            bkgerr.SetBinContent(i + 1, v / bkg.GetBinContent(i + 1))
        bkgerr.Sumw2()

    sig = ROOT.TH1F('sig', 'sig', *binning)
    for i, v in enumerate(bindata['sig']):
        sig.SetBinContent(i + 1, v)
    sig.Sumw2()
    f.Write()


def remove_results_dir():
    try:
        path_results_dir = os.path.join(os.getcwd(), 'results')
        shutil.rmtree(path_results_dir)
    except OSError:
        pass


def time_RunFixedScan():

    infile = ROOT.TFile.Open(
        './results/example_combined_GaussExample_model.root')

    start_time = time.time()
    workspace = infile.Get('combined')
    data = workspace.data('obsData')
    sbModel = workspace.obj('ModelConfig')
    poi = sbModel.GetParametersOfInterest().first()

    sbModel.SetSnapshot(ROOT.RooArgSet(poi))

    bModel = sbModel.Clone()
    bModel.SetName('bonly')
    poi.setVal(0)
    bModel.SetSnapshot(ROOT.RooArgSet(poi))

    ac = ROOT.RooStats.AsymptoticCalculator(data, bModel, sbModel)
    ac.SetPrintLevel(10)
    ac.SetOneSided(True)
    ac.SetQTilde(True)
    calc = ROOT.RooStats.HypoTestInverter(ac)
    calc.RunFixedScan(1, 1, 1)

    total_wall_time = time.time() - start_time
    return total_wall_time


def summary_stats(times):
    """
    Compute the mean, min, max, and standard deviation for the times of the runs

    Args:
        times: `list` of runs of times

    Returns:
        mean_time: `list` of mean times of all runs
        min_time: `list` of the minimum time out of all runs
        max_time: `list` of the maximum time out of all runs
        std_time: `list` of the standard deviation of the times in the runs
    """
    mean_time = np.mean(times, 0)
    min_time = np.amin(times, 0)
    max_time = np.amax(times, 0)
    std_time = np.std(times, 0)

    return mean_time.tolist(), min_time.tolist(), max_time.tolist(), std_time.tolist()


def run_test(n_bins):
    remove_results_dir()
    source = generate_source(n_bins)
    generate_data(source)
    call(['hist2workspace', 'config/example.xml'])
    return time_RunFixedScan()


def benchmark_ROOT_HF(n_bins, n_runs):
    """
    Benchmark the backends by timing multiple iterations of `runOnePoint` for
    different numbers of bins

    Args:
        backend: `pyhf.tensorlib` the current backend
        n_bins: `list` of numbers of bins to test
        n_runs: `int` the number of iterations to perform each test

    Returns:
        summary_stats(times): The mean, min, max, std of the timing tests
    """
    times = [[run_test(bins) for bins in n_bins] for _ in range(n_runs)]

    return summary_stats(times)


def run_benchmark(n_bins, n_runs=5, file_name='times_ROOT_HF'):
    times = {}
    times['n_bins'] = n_bins

    t_mean, t_min, t_max, t_std = benchmark_ROOT_HF(n_bins, n_runs)

    times['root'] = {
        'mean': t_mean,
        'min': t_min,
        'max': t_max,
        'std': t_std
    }

    with open(file_name + '.json', 'w') as outfile:
        json.dump(times, outfile)


if __name__ == '__main__':
    gROOT.SetBatch(ROOT.kTRUE)

    # n_bins = [1, 10, 50, 100, 200, 500, 800, 1000]
    n_bins = [1, 10, 50, 100, 200]

    run_benchmark(n_bins)
