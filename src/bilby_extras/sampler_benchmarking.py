#!/usr/bin/env python3

import copy
import logging

import argcomplete
import bilby
import configargparse as cfg
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
        A Bilby PriorDict object encoding the prior data from the .prior file
    """
    # setup dict
    prior_dict = bilby.core.prior.PriorDict()
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


def parser():
    """
    Parser config and input args to generate test parameters

    Returns
    ------------
    arguments
        A cfgargparse object
    """
    parser = cfg.ArgumentParser(
        config_file_parser_class=cfg.ConfigparserConfigFileParser
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
        type=dict,
        default={},
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
        type=dict,
        default={"nlive": 1000},
        help="Kwargs to pass to the sampler",
    )
    parser_prior = parser.add_argument_group(
        title="Prior Arguments",
        description="Arguments to pass to the prior - preferably just the .prior file",
    )
    parser_prior.add_argument(
        "--prior-dict",
        type=str,
        help="The path to the .prior file from which the prior dict is assembled",
    )
    parser_prior.add_argument(
        "--prior-extra-dict",
        type=dict,
        default={},
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
        type=dict,
        default={},
        help="The parameters to injection, for the given model, overrides injection file settings",
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


def gaussian_noise(noise_std, duration, srate):
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
    return np.random.normal(0, noise_std, duration * srate)


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
)
