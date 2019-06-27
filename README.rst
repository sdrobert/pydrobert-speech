[![Build Status](https://travis-ci.com/sdrobert/pydrobert-speech.svg?branch=master)](https://travis-ci.com/sdrobert/pydrobert-speech)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

pydrobert-speech
================

What is it?
-----------

A library for converting audio signals to speech features, likely for speech
recognition.

Layout
------

``pydrobert.speech`` contains the following important submodules

1. ``pre`` : audio -> audio. Preprocess audio signals.
2. ``scale`` : produces scaling functions for placing filter banks in time.
   E.g. Mel-scale, Bark-scale, or uniform.
3. ``filters`` : scale -> filter bank. Produces factories of filters
   (e.g. Gabor), positioned according to input scale. The
   factory can generate the same filter in the frequency or time domain.
   ``filters`` also contains windowing functions, e.g.    Hann.
4. ``compute`` : audio, filter bank, window -> features. Produces features,
   given a filter bank and a signal. The object in ``compute`` controls the
   conversion. The standard, Short-Time-Fourier-Transform-based filter bank
   representation is, for example, ``ShortTimeFourierTransformFrameComputer``.
5. ``post`` : features -> features. Post-processing features, e.g. CMVN.

These submodules naturally form a speech processing pipeline.

Pipeline construction
---------------------

Most objects in the library inherit from ``AliasedFactory``. Parent classes
inheriting ``AliasedFactory`` can initialize children using configuration
hierarchies and aliases, e.g.

>>> mel_scale = pydrobert.speech.scales.MelScaling()
>>> gabor_bank = pydrobert.speech.filters.LinearFilterBank.from_alias(
...     'gabor', 'mel', high_hz=5000)
>>> stft_computer = pydrobert.speech.compute.FrameComputer.from_alias(
...     'stft', {'alias': 'gabor', 'scaling_function': 'mel', 'high_hz': 5000},
...      window_function='hann')

In the latter two lines, the first argument to ``from_alias`` specifies an
alias for a given child. For example, ``GaborFilterBank.aliases`` contains
``"gabor"``. The second argument to ``from_alias`` is passed as a positional
argument to the constructor of the child, and the third is passed as a keyword
argument. The positional arguments of the constructors of ``gabor_bank`` and
``stft_computer`` are ``scaling_function`` and ``bank``, which are expected to
be instances of ``AliasedFactory`` subclasses themselves. Hence, the object can
initialize the ``AliasedFactory`` arguments directly. In the above examples,
the positional arguments to ``from_alias`` are used to initialize values
identical to the previous lines' variables.

Most importantly, the ``AliasedFactory`` system allows for convenient
serialization of speech processing pipelines using JSON. JSON is used to
configure the pipeline for command-line usage.

Command-line
------------

The following commands are available via command-line:

- ``compute-feats-from-kaldi-tables`` (requires *pydrobert-kaldi*): Given
  ``WavData`` stored in a Kaldi table, write to another table the result of a
  speech processing pipeline

Other Utilities
---------------

- ``pydrobert.speech.vis`` contains functions for plotting various parts of
  the pipeline, including the filter frequency
  response and spectrogram-like representations of speech features. Requires
  *matplotlib*.
- ``pydrobert.speech.utils.read_signal`` is a very versatile function for
  reading signals from a variety of sources.
