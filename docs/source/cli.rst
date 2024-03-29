Command-Line Interface
======================

compute-feats-from-kaldi-tables
-------------------------------

::

  compute-feats-from-kaldi-tables -h
  usage: compute-feats-from-kaldi-tables [-h] [-v VERBOSE] [--config CONFIG] [--print-args PRINT_ARGS] [--min-duration MIN_DURATION] [--channel CHANNEL] [--preprocess PREPROCESS] [--postprocess POSTPROCESS] [--seed SEED] wav_rspecifier feats_wspecifier computer_config
  
  Store features from a kaldi archive in a kaldi archive
  
      This command is intended to replace Kaldi's (https://kaldi-asr.org/) series of
      "compute-<something>-feats" scripts in a Kaldi pipeline.
      
  
  positional arguments:
    wav_rspecifier        Input wave table rspecifier
    feats_wspecifier      Output feature table wspecifier
    computer_config       JSON file or string to configure a 'pydrobert.speech.compute.FrameComputer' object to calculate features with
  
  options:
    -h, --help            show this help message and exit
    -v VERBOSE, --verbose VERBOSE
                          Verbose level (higher->more logging)
    --config CONFIG
    --print-args PRINT_ARGS
    --min-duration MIN_DURATION
                          Min duration of segments to process (in seconds)
    --channel CHANNEL     Channel to draw audio from. Default is to assume mono
    --preprocess PREPROCESS
                          JSON list of configurations for 'pydrobert.speech.pre.PreProcessor' objects. Audio will be preprocessed in the same order as the list
    --postprocess POSTPROCESS
                          JSON list of configurations for 'pydrobert.speech.post.PostProcessor' objects. Features will be postprocessed in the same order as the list
    --seed SEED           A random seed used for determinism. This affects operations like dithering. If unset, a seed will be generated at the moment
  
  New in version 0.4.0: if ruamel.yaml is installed
  (https://yaml.readthedocs.io/en/latest/), JSON arguments will be parsed as YAML 1.2
  by default. As JSON is valid YAML 1.2, you can continue to use JSON for configurations.

signals-to-torch-feat-dir
-------------------------

::

  usage: signals-to-torch-feat-dir [-h] [--channel CHANNEL] [--preprocess PREPROCESS] [--postprocess POSTPROCESS] [--force-as {aiff,npz,hdf5,table,kaldi,soundfile,pt,ogg,npy,file,flac,sph,wav}] [--seed SEED] [--file-prefix FILE_PREFIX] [--file-suffix FILE_SUFFIX] [--num-workers NUM_WORKERS]
                                   [--manifest MANIFEST]
                                   map [computer_config] dir
  
  Convert a map of signals to a torch SpectDataSet
  
      This command serves to process audio signals and convert them into a format that can
      be leveraged by "SpectDataSet" in "pydrobert-pytorch"
      (https://github.com/sdrobert/pydrobert-pytorch). It reads in a text file of
      format
  
          <utt_id_1> <path_to_signal_1>
          <utt_id_2> <path_to_signal_2>
          ...
  
      computes features according to passed-in settings, and stores them in the
      target directory as
  
          dir/
              <file_prefix><utt_id_1><file_suffix>
              <file_prefix><utt_id_2><file_suffix>
              ...
  
      Each signal is read using the utility "pydrobert.speech.util.read_signal()", which
      is a bit slow, but very robust to different file types (such as wave files, hdf5,
      numpy binaries, or Pytorch binaries). A signal is expected to have shape (C, S),
      where C is some number of channels and S is some number of samples. The
      signal can have shape (S,) if the flag "--channels" is set to "-1".
  
      Features are output as "torch.FloatTensor" of shape "(T, F)", where "T" is some
      number of frames and "F" is some number of filters.
  
      No checks are performed to ensure that read signals match the feature computer's
      sampling rate (this info may not even exist for some sources).
      
  
  positional arguments:
    map                   Path to the file containing (<utterance>, <path>) pairs
    computer_config       JSON file or string to configure a pydrobert.speech.compute.FrameComputer object to calculate features with. If unspecified, the audio (with channels removed) will be stored directly with shape (S, 1), where S is the number of samples
    dir                   Directory to output features to. If the directory does not exist, it will be created
  
  options:
    -h, --help            show this help message and exit
    --channel CHANNEL     Channel to draw audio from. Default is to assume mono
    --preprocess PREPROCESS
                          JSON list of configurations for 'pydrobert.speech.pre.PreProcessor' objects. Audio will be preprocessed in the same order as the list
    --postprocess POSTPROCESS
                          JSON list of configurations for 'pydrobert.speech.post.PostProcessor' objects. Features will be postprocessed in the same order as the list
    --force-as {aiff,npz,hdf5,table,kaldi,soundfile,pt,ogg,npy,file,flac,sph,wav}
                          Force the paths in 'map' to be interpreted as a specific type of data. table: kaldi table (key is utterance id); wav: wave file; hdf5: HDF5 archive (key is utterance id); npy: Numpy binary; npz: numpy archive (key is utterance id); pt: PyTorch binary; sph: NIST
                          SPHERE file; kaldi: kaldi object; file: numpy.fromfile binary. soundfile: force soundfile processing.
    --seed SEED           A random seed used for determinism. This affects operations like dithering. If unset, a seed will be generated at the moment
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers simultaneously computing features. Should not affect determinism when used in tandem with --seed. '0' means all work is done on the main thread
    --manifest MANIFEST   If specified, a list of utterances which have already been computed will be stored in this file. Utterances already listed in the file will be not be computed. Useful for resuming computations after an unexpected termination
  
  New in version 0.4.0: if ruamel.yaml is installed
  (https://yaml.readthedocs.io/en/latest/), JSON arguments will be parsed as YAML 1.2
  by default. As JSON is valid YAML 1.2, you can continue to use JSON for configurations.

