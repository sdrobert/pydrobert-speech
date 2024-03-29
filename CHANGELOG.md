# Change Log

## v0.4.0

- CLI now quietly reads configs as YAML 1.2 via
  [ruamel.yaml](https://yaml.readthedocs.io/en/latest/) if installed. As JSON
  is valid YAML 1.2, it shouldn't break anything.
- [webdataset](https://github.com/webdataset) hook in `pydrobert.speech.util`.
- Python 3.12 support.
- `read_signal` can read from binary I/O.
- Added `--manifest` option to `signals-to-torch-feature-dir`.
- The `axis` argument of `PreProcessor` is deprecated.
- `Dither` adds normally-distributed noise rather than uniform noise.
- Added [pytorch](https://pytorch.org/) wrappers around computers under
  `pydrobert.speech.torch`. Preprocessors and STFTFrameComputer have bona
  fide PyTorch implementations; the rest just wrap the Numpy code.
- `signals-to-torch-feat-dir` uses the PyTorch implementations.

## v0.3.0

- Moved `AliasedFactory` and `alias_factory_subclass_from_arg` to `alias`
  submodule.
- Removed Python 3.6 support.
- Bug fix for issue #14 - Gammatone alpha coefficient wrong for ERB.
- Added `Stack` to `post`.
- Added `soundfile` decoding to `read_signal`.
- Removed catch-all condition for `read_signal` (when all else fails, try
  Kaldi and then `numpy.fromfile`). Breaks v0.2.0 behaviour. Needed for
  `soundfile`.
- Default for sphere decoding of u-law/a-law is now to convert to pcm-16 rather
  than to stay encoded.
- Removed `setup.py` (builds with `pyproject.toml` and `setup.cfg`).
- Removed conda recipe in prep for [conda-forge](https://conda-forge.org/).
- `version.py` -> `_version.py`
- Cleaned up documentation.

## v0.2.0

A considerable amount of refactoring occurred for this build, chiefly to get
rid of Python 2.7 support. While the functionality did not change much for this
version, we have switched from a `pkgutil`-style `pydrobert` namespace to
PEP-420-style namespaces. As a result, *this package is not
backwards-compatible with previous `pydrobert` packages!* Make sure that if any
of the following are installed, they exceed the following version thresholds:

- `pydrobert-param >0.2.0`
- `pydrobert-kaldi >0.5.3`
- `pydrobert-pytorch >0.2.1`

Miscellaneous other stuff:

- Type hints everywhere
- Shifted python source to `src/`
- Black-formatted remaining source
- Removed `future` dependency
- Shifted most of the configuration to `setup.cfg`, leaving only a shell
  in `setup.py` to remain compatible with Conda builds
- Added `pyproject.toml` for [PEP
  517](https://www.python.org/dev/peps/pep-0517/).
- `tox.ini` for TOX testing
- Switched to AppVeyor for CI
- Messed around with documentation a little bit
- Added changelog :D
