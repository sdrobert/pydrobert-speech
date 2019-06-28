Overview
========

This section provides an overview of how *pydrobert-speech* is organized so
that you can get your feature representation just right.

The input to *pydrobert-speech* are (acoustic) signals. The output are
features. We call the operator that transforms the signal to some feature
representation a computer. Operators that act on the signal and produce a
signal, like random dithering, are preprocessors, and operators that act on
features and produce features, like unit normalization, are postprocessors. The
latter two operators exist in the submodules ``pydrobert.speech.pre`` and
``pydrobert.speech.post`` and follow the usage pattern

>>> y = Op().apply(x)

A computer, which can be found in ``pydrobert.speech.compute``, will require
more complicated initialization. The standard feature representation, which is
a 2D time-log-frequency matrix of energy, derives from
``compute.LinearFilterBankFrameComputer``. It calculates coefficients over
uniform time slices (frames) using a bank of filters. Children of
``LinearFilterBankFrameComputer`` all have similar representations and all use
linear banks of filters, but can be computed in different ways. The classic
method of computation is the
``compute.ShortTimeFourierTransformFrameComputer``.

Banks of filters are derived from
``pydrobert.speech.filters.LinearFilterBank``. Children of the parent class,
such as ``filters.ComplexGammatoneFilterBank``, will decide on the shape of the
filters.

``LinearFilterBankFrameComputer`` instances compute coefficients at uniform
intervals in time. However, the distribution over frequencies is decided by
the distribution of filter frequency responses from the filter bank, which, in
turn, depends on a scaling function. Scaling functions can be found in
``pydrobert.speech.scales``, such as ``scales.MelScaling``. Scaling functions
transform the frequency domain into some other real domain. In *that* domain,
filter frequency bandwidths are distributed uniformly which, when translated
back to the frequency domain, could be quite non-uniform.

In sum, you build a computer by first choosing a scale from
``pydrobert.speech.scales``. You then pass that as an argument to a filter
bank that you've chosen from ``pydrobert.speech.filters``. Finally, you past
that as an argument to your computer of choice. For example:

>>> from pydrobert.speech import *
>>> scale = scales.MelScaling()
>>> bank = filters.ComplexGammatoneFilterBank(scale)
>>> computer = compute.ShortTimeFourierTransformFrameComputer(bank)
>>> # preprocess the signal
>>> feats = computer.compute_full(signal)
>>> # postprocess the signal

This is a bit different from the syntax described in the ``README``. There, we
use aliases. Aliases are a simple mechanism for unpacking hierarchies of
parameters, such as the hierarchy between these computers, filter banks, and
scales. We can streamline the above initialization as

>>> computer = compute.ShortTimeFourierTransformFrameComputer(
...     {"name": "tonebank", "scaling_function": "mel"})

or even

>>> computer = compute.FrameComputer.from_alias("stft",
...     {"name": "tonebank", "scaling_function": "mel"})

The dictionaries are merely keyword argument dictionaries with the special key
``"name"`` or ``"alias"`` referring to an alias of the subclass you wish to
initialize (unless you just pass a string, at which point it's considered the
alias with no arguments). Aliases are listed in each subclass' ``alias`` class
member. Besides for brevity, aliases provide a principled way of storing
hierarchies on disk via JSON. Thus, it's possible to access most of
*pydrobert-speech*'s flexibility from the provided command-line hooks.

Finally, there are some visualization functions in the ``pydrobert.speech.vis``
module (requires ``matplotlib``), some extensions to *pydrobert-kaldi* data
iterators in ``pydrobert.speech.corpus``, and command line implementations
in ``pydrobert.speech.command_line``.
