import ROOT
ROOT.RooMsgService.instance().setGlobalKillBelow(5)
import itertools  # for fast looping
import time  # for timing loop


def sample_func_histogram(shape_function, n_events=1, n_bins=10, range_low=0, range_high=1, hist_name='data_hist'):
    """
    Sample a function. create an pdf from it, then histogram it

    Args:
        shape_function (TF1): The mathematical form of the function to be sampled
        n_events (int): The number of points to sample
        range_low (int, float): The lower bound on the sampling range
        range_high (int, float): The upper bound on the sampling range
        n_bins (int): The number of bins in the generated histogram
        hist_name (str): The name of the histogram object
    Returns:
        hist (TH1): The generated histogram of n_events
    """
    x = ROOT.RooRealVar('x', 'x', range_low, range_high)
    # Get the formula string of the TF1 with the parameters evaluated
    formula = str(shape_function.GetFormula().GetExpFormula("P"))
    pdf = ROOT.RooGenericPdf('mypdf', 'mypdf', formula, ROOT.RooArgList(x))
    data = pdf.generate(ROOT.RooArgSet(x), n_events)

    roo_data_hist = ROOT.RooDataHist(hist_name, "", ROOT.RooArgSet(x), data)
    hist = roo_data_hist.createHistogram(
        hist_name, x, ROOT.RooFit.Binning(n_bins, range_low, range_high))
    hist.Sumw2(False)
    hist.Sumw2()
    return hist


def make_sample(sample_name="sample", sample_histogram=None, uncertainty_down=None, uncertainty_up=None,
                shape_up_hist=None, shape_down_hist=None, has_theory_norm=False):
    """
    Construct a HistFactory.Sample()

    Args:
        sample_name (str): The name of the signal sample
        sample_histogram: A TH1 object that encodes the shape of the sample
        uncertainty_down (float): The downward normalisation systematic uncertainty (1.0 - uncertainty_down)
        uncertainty_up (float): The upward normalisation systematic uncertainty (1.0 + uncertainty_up)
        shape_up_hist: A TH1 object that encodes the updward sample shape uncertainty
        shape_down_hist: A TH1 object that encodes the downward sample shape uncertainty
        has_theory_norm (bool): Is the sample normalized by theory or not. Default is False.
    Returns:
        sample (HistFactory.Sample()): The constructed sample. Ex: signal sample, background sample
    """
    sample = ROOT.RooStats.HistFactory.Sample(sample_name)
    ROOT.SetOwnership(sample, False)
    sample.SetNormalizeByTheory(has_theory_norm)
    sample.SetHisto(sample_histogram)
    #
    sample.AddNormFactor("SigXsecOverSM", 1, 0, 3)
    # Add normalisation systematic uncertainty
    if uncertainty_down and uncertainty_up:
        sample.AddOverallSys(sample_name + "_uncertainty",
                             1.0 - uncertainty_down, 1.0 + uncertainty_up)
    # Add signal shape systematic uncertainty
    if shape_up_hist and shape_down_hist:
        shape_systematic = ROOT.RooStats.HistFactory.HistoSys(
            sample_name + "_shape_sys")
        shape_systematic.SetHistoHigh(shape_up_hist)
        shape_systematic.SetHistoLow(shape_down_hist)
        sample.AddHistoSys(shape_systematic)
    return sample


def make_channel(channel_name="channel", channel_data=None, channel_samples=None):
    """
    Make a channel with data and associated samples

    Args:
        channel_name (str): The name of the channel.
            Example: If this channel represents a signal region, the name might be "SR"
        channel_data (TH1): The data histogram that describes the channel
        channel_samples (list of HistFactory.Sample()): The samples associated with the channel that describe
                                                        the model for the channel

    Returns:
        channel (HistFactor.Channel()): The channel object
    """
    channel = ROOT.RooStats.HistFactory.Channel(channel_name)
    ROOT.SetOwnership(channel, False)
    channel.SetData(channel_data)
    channel.SetStatErrorConfig(0.05, "Poisson")
    if channel_samples:
        for sample in channel_samples:
            channel.AddSample(sample)
    return channel


def make_model(n_evnets, n_bins, channels, POI, workspace_name=None, workspace_save_path=None):
    """
    Args:
        n_events: The number of events
        n_bins: The numbe of bins
        channels (dictionary of HistFactory.Channel()): The channels in the model with associated samples
        # n_sig_comp: The number of signal components (automatically determined from the channels)
        # n_bkg_comp: The number of background components (automatically
        # determined from the channels)
        POI (list of str): The parameters of interest
        # n_nuisance: The number of nuisance parameters (automatically
        # determined from the samples)
        workspace_name (str): The name of the resulting RooStats.RooWorkspace()
        workspace_save_path (str): The path and filename (sans ".root") to save the workspace
    Returns:
        workspace (RooStats.RooWorkspace): The workspace for the measruement
        model (RooStats.RooSimultaneous): The model generated from the workspace
    """
    measurement = ROOT.RooStats.HistFactory.Measurement(
        "measurement", "measurement")
    ROOT.SetOwnership(measurement, False)
    for parameter in POI:
        measurement.SetPOI(parameter)
    # Set luminosity (arbitrary choices for now, which can be set later once
    # working)
    measurement.SetLumi(1.0)
    measurement.SetLumiRelErr(0.02)
    measurement.AddConstantParam("Lumi")
    # Add channels to measurement
    for channel in channels:
        measurement.AddChannel(channel)
    # make the factory
    factory = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast()
    if len(channels) == 1:
        workspace = factory.MakeSingleChannelModel(measurement, channels[0])
    else:
        workspace = factory.MakeCombinedModel(measurement)

    if workspace_name:
        workspace.SetName(workspace_name)
    else:
        workspace.SetName("HistFactoryWorkspace")

    if workspace_save_path:
        workspace.writeToFile(workspace_save_path + ".root")

    if len(channels) == 1:
        model = workspace.pdf("model_" + channels[0].GetName())
    else:
        # 'simPdf' is the default name assigned
        model = workspace.pdf('simPdf')
    return workspace, model


def benchmark_fit(var, model, n_events, n_trials, verbose=False):
    """
    Fit the given model and return the time it take

    Args:
        var (RooAbsReal and inheritors): The variable with range over which to fit the model
        model (RooAbsPdf): The pdf of the model
        n_events (int): The number of data points to generate from the model
        n_trials (int): The number of times to fit the model
        verbose (bool):
            False: Don't print information
            True: Print information on fit
    Returns:
        mean_fit_time (float): The mean time it takes to fit the model over n_trials times
    Example:
        ws, model = make_model(n_events, n_bins, channels, POI)
        benchmark_fit(ws.var("obs_x_SR"), model, n_events, n_trials)
    """
    # ensure that the variable has been transformed to a RooArgSet
    if isinstance(var, ROOT.RooRealVar):
        arg_set = ROOT.RooArgSet(ROOT.RooArgList(var))
    elif isinstance(var, ROOT.RooArgList):
        arg_set = ROOT.RooArgSet(var)
    elif isinstance(var, ROOT.RooArgSet):
        arg_set = var

    start_time = time.time()
    for _ in itertools.repeat(None, n_trials):
        data = model.generate(arg_set, n_events)
        if (n_trials == 1 and verbose):
            model.fitTo(data, ROOT.RooFit.Save(False))
        else:
            model.fitTo(data, ROOT.RooFit.Save(False),
                        ROOT.RooFit.PrintLevel(-1))
    end_time = time.time()
    time_duration = end_time - start_time
    mean_fit_time = time_duration / n_trials
    if verbose:
        print("{} was fit {} times in {} seconds".format(
            model.GetName(), n_trials, time_duration))
        print("The average fit time is {} seconds".format(mean_fit_time))
    return mean_fit_time


def benchmark_fit_with_workspace(work_space, model, n_events, n_trials, verbose=False):
    """
    Fit the given model and return the time it takes

    Args:
        work_space (RooWorkspace): The work space for the model
        model (RooAbsPdf): The pdf of the model
        n_events (int): The number of data points to generate from the model
        n_trials (int): The number of times to fit the model
        verbose (bool):
            False: Don't print information
            True: Print information on fit
    Returns:
        mean_fit_time (float): The mean time it takes to fit the model over n_trials times
    Example:
        ws, model = make_model(n_events, n_bins, channels, POI)
        benchmark_fit_with_workspace(ws, model, n_events, n_trials)
    """
    # Default name given to a ModelConfig() object
    model_config = work_space.obj("ModelConfig")
    observable = model_config.GetObservables()
    return benchmark_fit(observable, model, n_events, n_trials, verbose)


def roodataset_to_hist(name, x, data, verbose=False):
    """
    Histogram a RooDataSet object

    Args:
        name (str): The name of the resultant histogram
        x (ROOT.RooRealVar): The variable on the x-axis of the histogram
        data (ROOT.RooDataSet): The data set to be histogramed
        verbose (bool):
            True: Print out information on the RooDataHist
            False: Don't print out information

    Returns:
        hist (TH1): The histogram made from the RooDataSet
    """
    roo_data_hist = ROOT.RooDataHist(name, "", ROOT.RooArgSet(x), data)
    if verbose:
        roo_data_hist.Print("v")
    hist = roo_data_hist.createHistogram(name, x)
    hist.Sumw2(False)
    hist.Sumw2()
    hist.SetMinimum(0)
    hist.SetMaximum(hist.GetBinContent(hist.GetMaximumBin()) * 1.1)
    return hist


def main():
    import sys
    import os
    # Don't require pip install to test out
    #sys.path.append(os.getcwd() + '/../src')
    sys.path.append(os.getcwd() + '/../')
    from dfgmark import histfactorybench as hfbench

    n_events = 1000
    n_bins = 1
    range_low = 0
    range_high = 10

    frac_sig = 0.1
    frac_bkg = 1 - frac_sig

    signal_hist = ROOT.TH1F("signal_hist", "signal_hist",
                            n_bins, range_low, range_high)
    signal_hist.SetBinContent(1, frac_sig * n_events)

    background_hist = ROOT.TH1F(
        "background_hist", "background_hist", n_bins, range_low, range_high)
    background_hist.SetBinContent(1, frac_bkg * n_events)

    data_hist = ROOT.TH1F("data_hist", "data_hist",
                          n_bins, range_low, range_high)
    data_hist.SetBinContent(1, n_events)

    samples = []
    samples.append(make_sample("signal", signal_hist, 0.01, 0.01))
    samples.append(make_sample("background", background_hist, 0.01, 0.01))

    channels = []
    SR = make_channel(channel_name="SR",
                      channel_data=data_hist, channel_samples=samples)
    channels.append(SR)

    POI = ["SigXsecOverSM"]
    ws, model = make_model(n_events, n_bins, channels, POI)

    benchmark_fit(ws.var("obs_x_SR"), model, n_events, n_events, verbose=True)

    c = ROOT.TCanvas()
    f = ws.var("obs_x_SR").frame()
    print(ws.var("obs_x_SR"))
    ws.data("obsData").plotOn(f)
    f.Draw()
    c.Draw()
    c.SaveAs("obsData_frame.pdf")
    print(ws.data("obsData"))

    roo_data_hist = ROOT.RooDataHist("h_data", "h_data", ROOT.RooArgSet(
        ws.var("obs_x_SR")), ws.data("obsData"))
    h_data = hfbench.roodataset_to_hist(
        "h_data", ws.var("obs_x_SR"), ws.data("obsData"), verbose=True)

    c.cd()
    h_data.Draw("E")
    c.Draw()
    c.SaveAs("hist_data.pdf")

    # Test
    f1 = ROOT.TF1("f1", "[0]*x + [1]", 0, 10)
    f1.SetParameters(1.5, 0.5)
    hist = sample_func_histogram(f1, 10000, 10, 0, 10)
    c.cd()
    hist.Draw()
    c.Draw()
    c.SaveAs("test.pdf")

if __name__ == '__main__':
    main()
