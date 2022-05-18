#!/home/richard.udall/.conda/envs/scattering_bilby_dev/bin/python3

import ast
import copy
import inspect
import logging
import os
import shutil

import bilby
import configargparse as cfg
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import DeltaFunction
from bilby.core.sampler import run_sampler
from bilby.gw.detector import (
    PowerSpectralDensity,
    get_interferometer_with_fake_noise_and_injection,
    load_data_by_channel_name,
)
from bilby.gw.detector.interferometer import Interferometer
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.scattering import slow_scattering_wrapper
from bilby.gw.waveform_generator import WaveformGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleScattering(object):
    @staticmethod
    def parse_args_and_config():
        # Setup args and parser - note ini is a positional arg, passed when called from the command line
        # All other args are internal to the config, but can also be command-line edited

        parser = cfg.ArgumentParser(
            config_file_parser_class=cfg.ConfigparserConfigFileParser
        )
        parser.add_argument(
            "ini",
            is_config_file=True,
            type=str,
            help="Pass the ini to interpret",
        )
        parser_job = parser.add_argument_group(
            title="Job Arguments",
            description="Arguments to govern the creation and description of the job",
        )
        parser_job.add_argument(
            "--ligo-user-name",
            type=str,
            default=os.environ["LIGO_USER_NAME"],
            help="LIGO User Name for condor",
        )
        parser_job.add_argument(
            "--ligo-accounting",
            type=str,
            default=os.environ["LIGO_ACCOUNTING"],
            help="LIGO Accounting for condor",
        )
        parser_job.add_argument(
            "--outdir",
            help="The directory to output to; can be absolute or relative",
        )
        parser_job.add_argument(
            "--label",
            help="The label to give the sampling process a la Bilby",
        )
        parser_prior = parser.add_argument_group(
            title="Prior Arguments", description="Arguments describing the prior"
        )
        parser_prior.add_argument(
            "--prior-file",
            type=str,
            help="The prior file to read for constructing a prior",
        )
        parser_prior.add_argument(
            "--prior-extra-dict",
            default="{}",
            help="Overrides for parameter:prior - preferentially update the .prior file",
        )
        parser_injection = parser.add_argument_group(
            title="Injection Arguments",
            description="Arguments governing the injection, if applicable",
        )
        parser_injection.add_argument(
            "--add-custom-model",
            default="()",
            help="A tuple of (path_to_source_for_import, shorthand_name, function_name)",
        )
        parser_injection.add_argument(
            "--injection-model",
            type=str,
            help=f"The name of the model: options are {_model_map}, or can be added",
        )
        parser_injection.add_argument(
            "--injection-file",
            type=str,
            default=None,
            help="The injection file, if an injection should be generated",
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
        parser_data = parser.add_argument_group(
            title="Data Arguments",
            description="Arguments governing making or reading background data",
        )
        parser_data.add_argument(
            "--channel",
            type=str,
            default="fake",
            help="The channel to read for strain, should include IFO:...\
            If this is absent or passed as fake, it will be assumed data should be generated",
        )
        parser_data.add_argument(
            "--trigger-time",
            type=float,
            help="The trigger time to center on, used for data fetch. \
        If no geocent time prior is passed it will also be used to construct a time prior",
        )
        parser_data.add_argument(
            "--sampling-rate",
            type=int,
            help="The sampling rate for the analysis",
        )
        parser_data.add_argument(
            "--duration",
            type=int,
            help="The length of the segment in integer seconds",
        )
        parser_data.add_argument(
            "--psd-file",
            type=str,
            help="The psd file to read in",
        )
        parser_data.add_argument(
            "--minimum-frequency",
            type=float,
            default=10,
            help="The minimum frequency to do likelihood over",
        )
        parser_data.add_argument(
            "--ifo-name",
            type=str,
            help="The IFO to study - if not passed, will attempt to infer from channel",
        )
        parser_data.add_argument(
            "--zero-noise",
            default="False",
            help="If True, generate with zero noise",
        )
        parser_data.add_argument(
            "--psd-gen-kwargs",
            default="{}",
            help="kwargs to pass to bilby.gw.load_data_by_channel_name for PSD generation,\
            including psd_duration and psd_start_time",
        )
        parser_likelihood = parser.add_argument_group(
            title="Likelihood Arguments",
            description="Arguments to use for computing the likelihood",
        )
        parser_likelihood.add_argument(
            "--likelihood-model",
            type=str,
            help=f"the data model to compute likelihoods with, options are {_model_map.keys()}",
        )
        parser_likelihood.add_argument(
            "--likelihood-fixed-kwargs",
            default="{}",
            help="Fixed kwargs which should not be sampled over, to pass as **kwargs to the model",
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
            default="{}",
            help="Kwargs to pass to the sampler",
        )
        parser_plot = parser.add_argument_group(
            title="Plotting Arguments",
            description="Arguments to go to the plotting functions for the data",
        )
        parser_plot.add_argument(
            "--q-value", default="20", help="The Q value to fix the spectrogram at"
        )
        arguments = parser.parse_args()
        return arguments

    @staticmethod
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

    @staticmethod
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
                    logger.info(f"Could not add line {line} to the injection args")
                    logger.exception("The exception was")
        return injection_dict

    @staticmethod
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
        injection_args = SampleScattering.read_injection_file(arguments.injection_file)
        injection_args.update(ast.literal_eval(arguments.injection_parameters_extra))
        return injection_args

    @staticmethod
    def make_prior_dict(arguments):
        """
        Helper for both reading prior file and updating with command line args

        Parameters
        ---------------
        arguments
            The output of parse()

        Returns
        --------------
        prior_dict
            Dict of prior keys and generators
        """
        prior_dict = SampleScattering.read_prior_file(arguments.prior_file)
        arguments.prior_extra_dict = ast.literal_eval(arguments.prior_extra_dict)
        for key, val in arguments.prior_extra_dict.items():
            prior_dict[key] = eval(val)
        return prior_dict

    @staticmethod
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
        # get the non-time named kwargs
        model_kwargs = [x for x in inspect.getargspec(model_function)[0] if x != "t"]
        # get their defaults
        model_kwarg_defaults = inspect.getargspec(model_function)[-1]
        # for named kwargs, if they aren't already in the prior, add them as a delta function at their default value
        for i, key in enumerate(model_kwargs):
            if key not in prior_dict.keys():
                prior_dict[key] = DeltaFunction(model_kwarg_defaults[i], name=key)
        return prior_dict

    def __init__(self, arguments):
        # copy the arguments object
        self.original_arguments = copy.deepcopy(arguments)

        # set all arguments values as class attributes
        for key, val in arguments.__dict__.items():
            setattr(self, key, val)

        # make a copy of the model_map, and add to it if requested
        self._model_map = copy.copy(_model_map)
        self.add_custom_model = ast.literal_eval(self.add_custom_model)
        if self.add_custom_model != ():
            self.add_model(
                self.add_custom_model[0],
                self.add_custom_model[1],
                self.add_custom_model[2],
            )
        # make prior dict
        self.prior_dict = self.make_prior_dict(self.original_arguments)
        self.prior_dict = self.sanitize_input_prior(
            self._model_map[self.likelihood_model],
            self.prior_dict,
        )

        self.zero_noise = ast.literal_eval(self.zero_noise)

        # set our specific definition of start time
        self.start_time = self.trigger_time - 3 * self.duration / 4
        return

    def add_model(self, source, shorthand_name, function_name):
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
        self._model_map[shorthand_name] = getattr(function_module, function_name)

    def construct_injection_fixed_kwargs(self):
        """
        Sets injection default kwargs and updates them with passed fixed kwargs
        """
        default_injection_fixed_kwargs = dict(
            ifo_name="L1",
            number_harmonics=1,
        )
        self.injection_fixed_kwargs = ast.literal_eval(self.injection_fixed_kwargs)
        default_injection_fixed_kwargs.update(dict(ifo_name=self.ifo_name))
        default_injection_fixed_kwargs.update(self.injection_fixed_kwargs)
        self.injection_fixed_kwargs = default_injection_fixed_kwargs

    def construct_likelihood_fixed_kwargs(self):
        """
        Sets likelihood defaault kwargs and updates them with passed fixed kwargs
        """
        self.likelihood_fixed_kwargs = ast.literal_eval(self.likelihood_fixed_kwargs)
        default_likelihood_fixed_kwargs = dict(
            ifo_name="L1",
            number_harmonics=1,
        )
        default_likelihood_fixed_kwargs.update(dict(ifo_name=self.ifo_name))
        default_likelihood_fixed_kwargs.update(self.likelihood_fixed_kwargs)
        self.likelihood_fixed_kwargs = default_likelihood_fixed_kwargs

    def setup_waveform_generator(self, model_name, model_kwargs):
        """
        Make a waveform generator for the data constraints and given model

        Parameters
        -----------
        model_name
            The name of the model (the key in _model_map)
        model_kwargs
            The fixed kwargs necessary to pass to the model
        """
        waveform_generator = WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_rate,
            time_domain_source_model=self._model_map[model_name],
            waveform_arguments=model_kwargs,
            parameter_conversion=None,
            start_time=self.start_time,
        )
        return waveform_generator

    def create_ifo(self):
        """
        Create the ifo object - either read frame or generate noise and inject

        Returns
        ---------
        self.ifo
            The Interferometer object containing the data
        """
        if self.channel == "fake":
            # The case that this is an injection run
            # Read in a PSD - could add default to aLigo?
            self.power_spectral_density = (
                PowerSpectralDensity.from_power_spectral_density_file(self.psd_file)
            )

            # make the injection args, since they are now necessary
            self.injection_args = self.make_injection_dict(self.original_arguments)

            # Setup and update injection waveform kwargs
            self.construct_injection_fixed_kwargs()

            # Get a WF generator for the injection
            self.injection_waveform_generator = self.setup_waveform_generator(
                self.injection_model,
                self.injection_fixed_kwargs,
            )

            # Make the Interferometer, with injection and generated noise
            self.ifo = get_interferometer_with_fake_noise_and_injection(
                self.ifo_name,
                self.injection_args,
                waveform_generator=self.injection_waveform_generator,
                sampling_frequency=self.sampling_rate,
                power_spectral_density=self.power_spectral_density,
                duration=self.duration,
                start_time=self.start_time,
                outdir=self.outdir,
                label=self.label,
                plot=True,
                save=False,
                zero_noise=self.zero_noise,
                raise_error=False,
            )
        else:
            # The case of reading in real data
            # Defaults for how to make a PSD, then update and separate out positional args
            psd_gen_default_kwargs = dict(
                psd_start_time=self.start_time - 32,
                psd_duration=64,
                roll_off=0.2,
                overlap=0,
            )
            psd_gen_default_kwargs.update(self.psd_gen_kwargs)
            self.psd_gen_kwargs = psd_gen_default_kwargs
            self.psd_start_time = self.psd_gen_kwargs.pop("psd_start_time")
            self.psd_duration = self.psd_gen_kwargs.pop("psd_duration")

            # Get the data, and make the PSD
            self.ifo = load_data_by_channel_name(
                self.channel,
                self.start_time,
                self.psd_start_time,
                self.duration,
                self.psd_duration,
                sampling_frequency=self.sampling_rate,
                outdir=self.outdir,
                **self.psd_gen_kwargs,
            )

        # Set our analysis minimum frequency, and save the data
        self.ifo.minimum_frequency = self.minimum_frequency
        self.ifo.to_pickle(
            self.outdir,
            label=self.label,
        )

        # Make a timeseries for plotting
        self.tseries_gwpy = self.ifo.strain_data.to_gwpy_timeseries()
        return self.ifo

    def load_ifo(self):
        """
        Reads in previously created interferometer object
        """
        if self.channel == "fake":
            # make the injection args, since they are used for the result plot
            self.injection_args = self.make_injection_dict(self.original_arguments)

            # Setup and update injection waveform kwargs
            self.construct_injection_fixed_kwargs()
        self.ifo = Interferometer.from_pickle(
            os.path.join(self.outdir, f"{self.label}.pkl")
        )

    def plot_tseries_data(self):
        """
        Make a timeseries plot of the data stream

        Returns
        ---------
        ax
            a Matplotlib ax object
        plot
            a Matplotlib fig object
        """
        fig = self.tseries_gwpy.plot(
            label="Data",
            marker="o",
            color="b",
        )
        ax = fig.gca()
        if self.channel == "fake":
            signal = self.injection_waveform_generator.time_domain_strain(
                parameters=self.injection_args,
            )["plus"]
            ax.plot(
                self.tseries_gwpy.times,
                signal,
                linestyle="--",
                color="r",
                label="Injected Signal",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Strain")
        ax.legend()
        fig.savefig(
            os.path.join(
                self.outdir,
                f"{self.label}_tseries_data.png",
            )
        )

        return ax, fig

    def plot_qscan_data(self):
        """
        Makes a qscan of the data stream

        Returns
        -----------
        ax
            The matplotlib axis object
        plot
            The matplotlib plot object
        """
        self.q_value = ast.literal_eval(self.q_value)
        qspecgram = self.tseries_gwpy.q_transform(qrange=[self.q_value, self.q_value])
        fig = qspecgram.plot(figsize=[8, 6])
        ax = fig.gca()
        # ax.set_epoch(0)
        ax.set_yscale("log")
        ax.set_xlabel("Time [seconds]")
        ax.set_ylim(self.minimum_frequency, 512)
        ax.grid(True, axis="y", which="both")
        fig.add_colorbar(cmap="viridis", label="Normalized energy", vmin=0, vmax=50)
        fig.savefig(
            os.path.join(
                self.outdir,
                f"{self.label}_qscan_data.png",
            )
        )

        return ax, fig

    def setup_likelihood(self):
        """
        Prepares the likelihood function given the
        """
        self.construct_likelihood_fixed_kwargs()
        self.likelihood_waveform_generator = self.setup_waveform_generator(
            self.likelihood_model,
            self.likelihood_fixed_kwargs,
        )
        self.likelihood = GravitationalWaveTransient(
            interferometers=[self.ifo],
            waveform_generator=self.likelihood_waveform_generator,
            time_reference=self.ifo_name,
        )

    def sample(self):
        """
        Runs the sampler for this data and likelihood

        Returns
        ---------
        self.result
            The completed result object
        """
        if "injection_args" not in self.__dict__.keys():
            self.injection_args = None

        default_sampler_kwargs = dict(
            sample="rwalk",
            npoints=1000,
            dlogz=0.1,
            proposals=["diff", "walk", "snooker"],
        )
        default_sampler_kwargs.update(self.sampler_kwargs)
        self.sampler_kwargs = default_sampler_kwargs
        self.result = run_sampler(
            self.likelihood,
            self.prior_dict,
            sampler=self.sampler,
            outdir=self.outdir,
            label=self.label,
            resume=True,
            injection_parameters=self.injection_parameters,
            **self.sampler_kwargs,
        )
        return self.result

    def produce_corner(self):
        """
        Produces the cornerplot for a completed result
        """
        self.result.plot_corner()


_model_map = dict(
    slow_scattering=bilby.gw.scattering.slow_scattering_wrapper,
    # fast_scattering = bilby.gw.scattering.fast_scattering_wrapper,
)


def setup_job():
    """
    Method for constructing dag and setting up data for scattering inference job
    """
    arguments = SampleScattering.parse_args_and_config()

    # imports
    import configparser

    from glue import pipeline

    # if local directory make global path, if global use it
    if arguments.outdir[-1] != "/":
        arguments.outdir = os.path.join(os.getcwd(), arguments.outdir)

    ligo_accounting_group = arguments.ligo_accounting
    ligo_accounting_user = arguments.ligo_user_name

    # make sure the run doesn't already exist, and make the directory
    assert not os.path.isdir(arguments.outdir)
    os.mkdir(arguments.outdir)

    # copy over prior, reassign path to copy
    prior_in_rundir = os.path.join(arguments.outdir, f"{arguments.label}.prior")
    shutil.copy(arguments.prior_file, prior_in_rundir)
    arguments.prior_file = prior_in_rundir

    # copy over injection if necessary, reassign path to copy
    if arguments.channel == "fake":
        injection_in_rundir = os.path.join(
            arguments.outdir, f"{arguments.label}.injection"
        )
        shutil.copy(arguments.injection_file, injection_in_rundir)
        arguments.injection_file = injection_in_rundir

    # write config in clean way
    config_in_rundir = os.path.join(arguments.outdir, f"{arguments.label}.cfg")
    arguments_dict = arguments.__dict__
    arguments_dict.pop("ini")
    write_dict = copy.deepcopy(arguments_dict)
    for key, value in arguments_dict.items():
        write_dict[key.replace("_", "-")] = str(write_dict.pop(key))
    parser = configparser.ConfigParser()
    parser["Arguments"] = write_dict
    with open(config_in_rundir, "w") as f:
        parser.write(f)

    # setup the submit file
    scattering_dag = pipeline.CondorDAG(
        log=os.path.join(arguments.outdir, "dag_scattering_PE.log")
    )
    scattering_dag.set_dag_file(os.path.join(arguments.outdir, "dag_scattering_PE"))

    # Make the sub and node
    sampler_exe = shutil.which("scattering_run")
    sampler_job = pipeline.CondorDAGJob(universe="vanilla", executable=sampler_exe)
    sampler_job.set_log_file(os.path.join(arguments.outdir, "sampler.log"))
    sampler_job.set_stdout_file(os.path.join(arguments.outdir, "sampler.out"))
    sampler_job.set_stderr_file(os.path.join(arguments.outdir, "sampler.err"))
    sampler_job.add_condor_cmd("accounting_group", ligo_accounting_group)
    sampler_job.add_condor_cmd("accounting_group_user", ligo_accounting_user)
    sampler_job.add_condor_cmd("request_memory", "10000")
    sampler_job.add_condor_cmd("request_disk", "10000")
    sampler_job.add_condor_cmd("notification", "never")
    sampler_job.add_condor_cmd("initialdir", arguments.outdir)
    sampler_job.add_condor_cmd("get_env", "True")
    sampler_args = f"{config_in_rundir}"
    sampler_job.add_arg(sampler_args)
    sampler_job.set_sub_file(os.path.join(arguments.outdir, "sampler.sub"))
    sampler_node = sampler_job.create_node()
    sampler_node.set_retry(5)

    # Construct the dag and write files
    scattering_dag.add_node(sampler_node)
    scattering_dag.write_sub_files()
    scattering_dag.write_dag()

    # Make the data, save, and plot
    Scattering = SampleScattering(arguments)
    Scattering.create_ifo()
    Scattering.plot_tseries_data()
    Scattering.plot_qscan_data()

    # Launch the condor job
    # os.system(f"condor_submit_dag {os.path.join(arguments.outdir, 'dag_scattering_PE.dag')}")


def perform_inference():
    """
    Method for performing scattering inference
    """

    # Setup scattering object
    arguments = SampleScattering.parse_args_and_config()
    Scattering = SampleScattering(arguments)

    # Load data, setup likelihood, run, make corner
    Scattering.load_ifo()
    Scattering.setup_likelihood()
    Scattering.Sample()
    Scattering.produce_corner()

    return
