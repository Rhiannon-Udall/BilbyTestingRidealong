#!/usr/bin/env python3

import copy

import argcomplete
import bilby
import configargparse as cfg
import numpy as np


def parser():
    """
    Parser config and input args to generate test parameters

    Returns
    ------------
    opts
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
        help="The type of noise: options are zero, gaussian. Noise kwargs should match type.",
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
        title="Injection Arguments", description="Arguments to form the injection",
    )
    parser_injection.add_argument(
        "--injection-model",
        type=str,
        default="linear",
        help=f"The data model to inject with. Options are {_model_map.keys}",
    )
    parser_injection.add_argument(
        "--injection-parameters",
        type=dict,
        default={},
        help="The parameters to injection, for the given model",
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


def linear_model(t, m, b):
    return m * t + b


def sine_model(t, omega, phi, amplitude):
    return amplitude * np.sin(omega * t + phi)


def gaussian_noise(noise_std, duration, srate):
    return np.random.normal(0, noise_std, duration * srate)


def zero_noise(duration, srate):
    return np.zeros(duration * srate)


_noise_map = dict(
    gaussian=gaussian_noise,
    zero=zero_noise,
)

_model_map = dict(
    linear=linear_model,
    sine=sine_model,
)
