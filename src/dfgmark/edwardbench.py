import numpy as np
import tensorflow as tf
import edward as ed
# specific modules
from edward.models import Normal
from edward.models import Categorical, Mixture


def make_channel(members):
    """
    Create the equivalent of a HistFactory Channel(). This is a composite model of p.d.f.s
    whose fractional weights sum to unity.

    Args:
        members (dict of ed.models): The p.d.f.s that will comprise the channel model
        along with their relative fractional weights. The dict should have elements of
            {'pdf_name':[fractional_weight, pdf]}

    Returns:
        channel (ed.models.Mixture): The resulting mixture model from the weighted
        combinations of the members
    """
    fracs = [v[0] for v in members.values()]
    assert 1. == sum(fracs),\
        "The sum of the p.d.f. samples fractional weights must be unity.\n\
    1 != {0}".format(sum(fracs))

    from edward.models import Categorical, Mixture

    cat = Categorical(probs=fracs)
    components = [v[1] for v in members.values()]

    channel = Mixture(cat=cat, components=components)

    return channel


def sample_model(model_template, n_samples):
    """
    Make n_sample observations of an Edward model

    Args:
        model_template (edward.models): An Edward model (a sample_shape is not required)
        n_samples (int): The number of observation of the model to make

    Returns:
        model (edward.models): An Edward model with sample_shape=n_samples
        samples (np.ndarray): An array of n_samples sampled observation of model
    """
    model = model_template.copy(sample_shape=n_samples)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = sess.run(model)
    return model, samples


def fit_model(model, observations, POI, fit_type='mle'):
    """
    Perform a fit of the model to data

    Args:
        model (ed.models class): An Edward model
        observations (np.ndarray): Data to fit the model to
        POI (dict): Parameters of interest to return fit results on
        fit_type (str): The minimization technique used

    Returns:
        fit_result (dict): A dict of the fitted model parameters of interest
    """
    # observations is an ndarray of (n_observations, d_features)
    # model and data (obsevations) need to have the same size
    assert model.get_shape() == observations.shape,\
        "The model and observed data features must be of the same shape.\n\
    The model passed has shape {0} and the data passed have shape (n_observations, d_features) = {1}".format(
        model.get_shape(), observations.shape)

    fit_type = fit_type.lower()
    if fit_type == 'mle':
        # http://edwardlib.org/api/ed/MAP
        fit = ed.MAP({}, data={model: observations})
    else:
        fit = ed.MAP({}, data={model: observations})  # default to mle
    fit.run()

    sess = ed.get_session()

    fit_result = {}
    for poi in POI:
        fit_result[poi] = sess.run(POI[poi])
    return fit_result


def main():
    import sys
    import os
    # Don't require pip install to test out
    #sys.path.append(os.getcwd() + '/../src')
    sys.path.append(os.getcwd() + '/../')
    from dfgmark import edwardbench as edbench
    import matplotlib.pyplot as plt

    N = 10000

    mean1 = tf.Variable(0., name='mean1')
    mean2 = tf.Variable(3., name='mean2')

    mu1 = Normal(loc=mean1, scale=1.)
    mu2 = Normal(loc=mean2, scale=1.)
    frac_1 = 0.4
    frac_2 = 1 - frac_1
    cat = Categorical(probs=[frac_1, frac_2])
    components = [mu1, mu2]
    # Gaussian mixture model
    model_template = Mixture(cat=cat, components=components)

    model, samples = edbench.sample_model(model_template, N)

    POI = {'mean1': mean1,
           'mean2': mean2}
    fit_result = edbench.fit_model(model, samples, POI)
    print(fit_result)

    plt.hist(samples, bins=50, range=(-3.0, 9.0))
    plt.show()

if __name__ == '__main__':
    main()
