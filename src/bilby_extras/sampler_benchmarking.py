#!/usr/bin/env python3

import logging
import os

import bilby
import configargparse as cfg
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_prior_file(prior_file):
    """
    Reads a bilby prior file, producing a PriorDict object

    Parameters
    ----------
    prior_file
        The file path for a .prior file

    Returns
    -----------
    prior_dict
        A dictionary encoding the prior data from the .prior file
    """
    # setup dict
    prior_dict = {}
    # read file
    with open(prior_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        # ignore comments and blank lines
        if line[0] == "#" or line[0] == "\n":
            continue
        else:
            try:
                # split up key and arg, evaluate the arg
                key = line.split("=")[0].strip()
                args = "=".join(line.split("=")[1:])
                prior_dict[key] = eval(args)
            # except out of malformed lines
            except SyntaxError:
                logger.info(f"Could not add line {line} to the PriorDict")
                logger.exception("The exception was")
    return prior_dict


def read_injection_file(injection_file):
    """
    Reads an injection file for the input injection data

    Parameters
    ----------
    injection_file
        The path to the file to read

    Returns
    -----------
    injection_dict
        A dictionary of injection parameters
    """
    # setup dict
    injection_dict = {}
    # read file
    with open(injection_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        # ignore comments and blank lines
        if line[0] == "#" or line[0] == "\n":
            continue
        else:
            try:
                # split up key and arg, evaluate the arg
                key = line.split("=")[0].strip()
                args = "=".join(line.split("=")[1:])
                injection_dict[key] = eval(args)
            # except out of malformed lines
            except SyntaxError:
                logger.info(f"Could not add line {line} to the PriorDict")
                logger.exception("The exception was")
    return injection_dict


def add_model(source, shorthand_name, function_name):
    """
    A method for adding a new model for data generation, from a .py file

    Parameters
    --------------
    source
        The path to the .py file where the function may be found
    shorthand_name
        The name to use in calling the model (should be the same as the name in the .cfg)
    function_name
        The name of the function to map in

    Modifies
    ---------------
    _model_map
        Adds the function with key shorthand_name
    """
    # import the module from the path
    import importlib.util

    spec = importlib.util.spec_from_file_location("import_module", source)
    function_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(function_module)
    # add the specific function to the model map
    _model_map[shorthand_name] = getattr(function_module, function_name)


def parse():
    """
    Parser config and input args to generate test parameters

    Returns
    ------------
    arguments
        A configargparse object
    """
    parser = cfg.ArgumentParser(
        config_file_parser_class=cfg.ConfigparserConfigFileParser
    )
    parser.add_argument(
        "config",
        is_config_file=True,
        help="The config file to read",
    )
    parser_data = parser.add_argument_group(
        title="Data Arguments",
        description="Arguments for the creation and description of the data",
    )
    parser_data.add_argument(
        "--duration",
        type=int,
        default=8,
        help="The duration of the timeseries data",
    )
    parser_data.add_argument(
        "--srate",
        type=int,
        default=128,
        help="The sample rate of the timeseries data",
    )
    parser_data.add_argument(
        "--noise-type",
        type=str,
        default="zero",
        help=f"The type of noise: options are {_noise_map}. Noise kwargs should match type.",
    )
    parser_data.add_argument(
        "--noise-kwargs",
        default="{}",
        help="Kwargs for noise function, not including srate and duration",
    )
    parser_sampler = parser.add_argument_group(
        title="Sampler Arguments",
        description="Arguments to pass to the sampler",
    )
    parser_sampler.add_argument(
        "--sampler",
        type=str,
        default="dynesty",
        help="The sampler to use e.g. Dynesty",
    )
    parser_sampler.add_argument(
        "--sampler-kwargs",
        default="{'nlive': 1000}",
        help="Kwargs to pass to the sampler",
    )
    parser_prior = parser.add_argument_group(
        title="Prior Arguments",
        description="Arguments to pass to the prior - preferably just the .prior file",
    )
    parser_prior.add_argument(
        "--prior-file",
        type=str,
        help="The path to the .prior file from which the prior dict is assembled",
    )
    parser_prior.add_argument(
        "--prior-extra-dict",
        default="{}",
        help="Overrides for parameter:prior - preferentially update the .prior file",
    )
    parser_job = parser.add_argument_group(
        title="Job Arguments",
        description="Arguments regarding the job itself, e.g. output directory",
    )
    parser_job.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="The directory for the analysis, defaults to cwd",
    )
    parser_job.add_argument(
        "--label",
        type=str,
        default="mylabel",
        help="The label for the bilby analysis",
    )
    parser_injection = parser.add_argument_group(
        title="Injection Arguments",
        description="Arguments to form the injection",
    )
    parser_injection.add_argument(
        "--injection-model",
        type=str,
        default="linear",
        help=f"The data model to inject with. Options are {_model_map.keys()}",
    )
    parser_injection.add_argument(
        "--injection-file",
        type=str,
        help="The file to read injection parameters from",
    )
    parser_injection.add_argument(
        "--injection-parameters-extra",
        default="{}",
        help="The parameters to injection, for the given model, overrides injection file settings",
    )
    parser_injection.add_argument(
        "--injection-fixed-kwargs",
        default="{}",
        help="Fixed kwargs which should not be sampled over, to pass as **kwargs to the model",
    )
    parser_injection.add_argument(
        "--add-custom-model",
        default="()",
        help="A tuple of (path_to_source_for_import, shorthand_name, function_name)",
    )
    parser_likelihood = parser.add_argument_group(
        title="Likelihood Arguments",
        description="Arguments to use when evaluating the likelihood for the model",
    )
    parser_likelihood.add_argument(
        "--estimate-sigma",
        action="store_true",
        help="Whether the standard deviation of the noise should also be estimated \
        (assuming noise is drawn from gaussian distribution)",
    )
    parser_likelihood.add_argument(
        "--likelihood-model",
        type=str,
        default="linear",
        help=f"the data model to compute likelihoods with, options are {_model_map.keys()}",
    )
    parser_likelihood.add_argument(
        "--likelihood-fixed-kwargs",
        default="{}",
        help="Fixed kwargs which should not be sampled over, to pass as **kwargs to the model",
    )
    arguments = parser.parse_args()
    return arguments


def linear_model(t, m, b):
    """
    Linear data for the time range

    Parameters
    -------------
    t
        The time array to evaluate over
    m
        The line's slope
    b
        The line's y intercept

    Returns
    --------------
    data
        The line over time
    """
    return m * t + b


def sine_model(t, omega, phi, amplitude):
    """
    Sinusoidal model over the time range

    Parameters
    -----------
    t
        The time array to evaluate over
    omega
        The angular frequency of the oscillation
    phi
        The phase of the oscillation
    amplitude
        The amplitude of the oscillation

    Returns
    ----------
    data
        The data for the sinusoid over time
    """
    return amplitude * np.sin(omega * t + phi)


def damped_sinusoid(t, omega, phi, amplitude_initial, damping_time):
    """
    Sinusoidal model over the time range

    Parameters
    -----------
    t
        The time array to evaluate over
    omega
        The angular frequency of the oscillation
    phi
        The phase of the oscillation
    amplitude_initial
        The amplitude at the beginning of the oscillation
    damping_time
        The exponential damping time of the sinusoid (1 / the complex part of the angular frequency)

    Returns
    ----------
    data
        The data for the damped sinusoid over time
    """
    sinusoid = amplitude_initial * np.sin(omega * t + phi)
    damping_envelope = np.exp(-t / damping_time)
    return sinusoid * damping_envelope


def gaussian_noise(duration, srate, sigma):
    """
    Produces Gaussian noise (flat PSD) over time

    Parameters
    ------------
    noise_std
        The standard deviation of the Gaussian generating the noise
    duration
        The duration of the signal
    srate
        The sampling rate for the signal

    Returns
    ------------
    data
        The noise realization over the time range
    """
    return np.random.normal(0, sigma, duration * srate)


def zero_noise(duration, srate):
    """
    Produces an empty array for the data over time, reflecting a zero noise case

    Parameters
    -------------
    duration
        The duration of the signal
    srate
        The sampling rate for the signal

    Returns
    --------------
    data
        An empty array of the correct length
    """
    return np.zeros(duration * srate)


_noise_map = dict(
    gaussian=gaussian_noise,
    zero=zero_noise,
)

_model_map = dict(
    linear=linear_model,
    sine=sine_model,
    damped_sine=damped_sinusoid,
)


def make_injection_dict(arguments):
    """
    Helper for both reading injection file and updating with command line args

    Parameters
    ---------------
    arguments
        The output of parse()

    Returns
    --------------
    injection_args
        Arguments to pass to the injection model
    """
    injection_args = read_injection_file(arguments.injection_file)
    injection_args.update(eval(arguments.injection_parameters_extra))
    return injection_args


def sanitize_input_prior(model_function, prior_dict):
    """
    When a function has many complex parameters and some are optional, set the unused ones to 0

    Parameters
    ---------------
    arguments
        The function used for the likelihood - should be from _model_map
    prior_dict
        The prior dict to update - should already have all .prior and cmd line args added

    Returns
    -------------
    prior_dict
        The prior_dict with all necessary extra parameters fixed
    """

    import inspect

    # get the non-time named kwargs
    model_kwargs = [x for x in inspect.getargspec(model_function)[0] if x != "t"]
    # get their defaults
    model_kwarg_defaults = inspect.getargspec(model_function)[-1]
    # for named kwargs, if they aren't already in the prior, add them as a delta function at their default value
    for i, key in enumerate(model_kwargs):
        if key not in prior_dict.keys():
            prior_dict[key] = bilby.core.prior.DeltaFunction(
                model_kwarg_defaults[i], name=key
            )
    return prior_dict


def initialize(arguments):
    """
    Setup the run directory, write data to a data file, and plot the data

    Parameters
    -------------
    arguments
        The output of parse()
    """
    # prepare run directory if necessary
    if not os.path.isdir(arguments.outdir):
        os.mkdir(arguments.outdir)

    # get injection arguments
    injection_args = make_injection_dict(arguments)

    arguments.injection_fixed_kwargs = eval(arguments.injection_fixed_kwargs)
    injection_args.update(arguments.injection_fixed_kwargs)

    # make time, noise, signal arrays - combine for data
    time = np.arange(0, arguments.duration, 1 / arguments.srate)
    noise = _noise_map[arguments.noise_type](
        arguments.duration, arguments.srate, **eval(arguments.noise_kwargs)
    )
    signal = _model_map[arguments.injection_model](time, **injection_args)
    data = signal + noise

    # plot the data
    fig, ax = plt.subplots()
    ax.plot(time, data, "o", label="data")
    ax.plot(
        time,
        signal,
        "--r",
        label="signal",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(f"{arguments.outdir}/{arguments.label}_data.png")

    # save the data
    np.save(f"{arguments.outdir}/{arguments.label}_data", data)

    return time, data


def run_sampler():
    """
    Runs the sampler for the data

    Inputs
    -------------------
    """
    # get arguments
    arguments = parse()

    # add a custom model if necessary
    arguments.add_custom_model = eval(arguments.add_custom_model)
    if arguments.add_custom_model != ():
        add_model(
            arguments.add_custom_model[0],
            arguments.add_custom_model[1],
            arguments.add_custom_model[2],
        )

    # if the run already exists load it, else make it
    if os.path.exists(f"{arguments.outdir}/{arguments.label}_data.npy"):
        data = np.load(f"{arguments.outdir}/{arguments.label}_data.npy")
        time = np.arange(0, arguments.duration, 1 / arguments.srate)
    else:
        time, data = initialize(arguments)

    # get injection arguments, make and update the prior dict according to config
    injection_args = make_injection_dict(arguments)
    prior_dict = read_prior_file(arguments.prior_file)
    arguments.prior_extra_dict = eval(arguments.prior_extra_dict)
    for key, val in arguments.prior_extra_dict.items():
        prior_dict[key] = eval(val)

    # determine if we will be doing estimate_sigma, and set the prior on sigma accordingly
    arguments.noise_kwargs = eval(arguments.noise_kwargs)
    if arguments.noise_type == "gaussian" and arguments.estimate_sigma:
        prior_dict["sigma"] = bilby.core.prior.Uniform(
            0,
            2 * np.abs(arguments.noise_kwargs["sigma"]),
        )

    prior_dict = sanitize_input_prior(
        _model_map[arguments.likelihood_model], prior_dict
    )

    # convert to PriorDict
    priors = bilby.core.prior.PriorDict(prior_dict)
    # choose likelihood method and arguments based on noise type and settings
    arguments.likelihood_fixed_kwargs = eval(arguments.likelihood_fixed_kwargs)
    if arguments.noise_type == "gaussian" and arguments.estimate_sigma:
        likelihood = bilby.likelihood.GaussianLikelihood(
            time,
            data,
            _model_map[arguments.likelihood_model],
            **arguments.likelihood_fixed_kwargs,
        )
    elif arguments.noise_type == "gaussian":
        likelihood = bilby.likelihood.GaussianLikelihood(
            time,
            data,
            _model_map[arguments.likelihood_model],
            sigma=arguments.noise_kwargs["sigma"],
            **arguments.likelihood_fixed_kwargs,
        )
    elif arguments.noise_type == "zero":
        likelihood = bilby.likelihood.GaussianLikelihood(
            time,
            data,
            _model_map[arguments.likelihood_model],
            sigma=0,
            **arguments.likelihood_fixed_kwargs,
        )
    else:
        logger.error("Cannot use Gaussian Likelihood with non-gaussian noise!")
        raise ValueError

    # get the sampler kwargs
    arguments.sampler_kwargs = eval(arguments.sampler_kwargs)

    # run the sampler, plot the final results
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=arguments.sampler,
        injection_parameters=injection_args,
        outdir=arguments.outdir,
        label=arguments.label,
        **arguments.sampler_kwargs,
    )
    result.plot_corner()
