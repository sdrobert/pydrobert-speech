import os
import tarfile
import wave
import json

from math import erf

import numpy as np
import pytest

from pydrobert.speech import util


@pytest.mark.parametrize("shift", [0, 1, 100, -100])
@pytest.mark.parametrize("dft_size", [1, 2, 51, 1000], ids=["l1", "l2", "l51", "l1000"])
@pytest.mark.parametrize("copy", [True, False], ids=["copy", "keep"])
@pytest.mark.parametrize("start_idx", [0, 1, -1], ids=["at", "after", "before"])
def test_circshift_fourier(shift, dft_size, start_idx, copy):
    start_idx %= dft_size
    zeros = np.random.randint(dft_size)
    X = 10 * np.random.random(dft_size - zeros) + 10j * np.random.random(
        dft_size - zeros
    )
    Xs = util.circshift_fourier(
        X.copy(), shift, start_idx=start_idx, dft_size=dft_size, copy=copy
    )
    X = np.roll(np.pad(X, (0, zeros), "constant"), start_idx)
    Xs = np.roll(np.pad(Xs, (0, zeros), mode="constant"), start_idx)
    assert len(X) == len(Xs)
    x = np.fft.ifft(X)
    xs = np.fft.ifft(Xs)
    assert np.allclose(np.roll(x, shift), xs)


@pytest.mark.parametrize("mu", [0, -1, 100])
@pytest.mark.parametrize("std", [0.1, 1, 10])
@pytest.mark.parametrize("do_scipy", [True, False])
def test_gauss_quant(mu, std, do_scipy):
    X = np.arange(1000, dtype=float) / 1000 - 0.5
    X /= X.std()
    X *= std / 2
    X += mu
    for x in X:
        p = 0.5 * (1 + erf((x - mu) / std / np.sqrt(2)))
        if do_scipy:
            pytest.importorskip("scipy.norm")
            x2 = util.gauss_quant(p, mu=mu, std=std)
        else:
            # because we don't give access to this if scipy is
            # installed, we have to access the private function
            x2 = util._gauss_quant_odeh_evans(p, mu=mu, std=std)
        assert np.isclose(x, x2, atol=1e-5)


@pytest.mark.parametrize("key", [True, False])
def test_read_table(temp_dir, key):
    kaldi = pytest.importorskip("pydrobert.kaldi.io")
    rxfilename = "ark:{}".format(os.path.join(temp_dir, "foo.ark"))
    key_1 = "lions"
    key_2 = "tigers"
    buff_1 = np.random.random((100, 10))
    buff_2 = np.random.random((1000, 2))
    with kaldi.open(rxfilename, "dm", "w") as table:
        table.write(key_1, buff_1)
        table.write(key_2, buff_2)
    if key:
        buff_3 = util.read_signal(rxfilename, dtype="dm", key=key_2)
        assert np.allclose(buff_2, buff_3)
    else:
        buff_3 = util.read_signal(rxfilename, dtype="dm")
        assert np.allclose(buff_1, buff_3)


@pytest.mark.parametrize("backend", ["wav", "scipy", "soundfile"])
@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
@pytest.mark.parametrize("sampwidth", [2, 4])
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_wave(temp_dir, backend, channels, sampwidth, from_file):
    rfilename = os.path.join(temp_dir, "foo.wav")
    if channels > 1:
        wave_buffer_1 = np.random.random((1000, channels)) * 1000
    else:
        wave_buffer_1 = np.random.random(1000) * 1000
    wave_buffer_1 = wave_buffer_1.astype("<i{}".format(sampwidth))
    wave_bytes = wave_buffer_1.tobytes("C")
    wave_file = wave.open(rfilename, "wb")
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(sampwidth)
    wave_file.setframerate(8000)
    wave_file.writeframes(wave_bytes)
    wave_file.close()
    if not from_file:
        rfilename = open(rfilename, "rb")
    if backend == "scipy":
        pytest.importorskip("scipy")
        wave_buffer_2 = util._scipy_io_read_signal(rfilename, None, None)
    elif backend == "soundfile":
        sf = pytest.importorskip("soundfile")
        if "WAV" not in sf.available_formats():
            pytest.skip("libsndfile cannot handle wav files")
        wave_buffer_2 = util._soundfile_read_signal(rfilename, None, None)
    else:
        wave_buffer_2 = util._wave_read_signal(rfilename, None, None)
    assert np.allclose(wave_buffer_1, wave_buffer_2)


@pytest.mark.parametrize(
    "name",
    [
        "123_1pcbe",
        "123_1pcle",
        "123_1ulaw",
        "123_2alaw",
        "123_2pcbe",
        "123_2pcle",
        "123_2ulaw",
    ],
)
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_sphere(name, from_file):
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    if name.endswith("alaw"):
        sph_file = os.path.join(audio_dir, name + ".sph")
    else:
        sph_file = os.path.join(audio_dir, name + "_shn.sph")
    wav_file = os.path.join(audio_dir, name + ".wav")
    assert os.path.isfile(sph_file)
    assert os.path.isfile(wav_file)
    wav = util.read_signal(wav_file)
    if from_file:
        force_as = None
    else:
        force_as = "sph"
        sph_file = open(sph_file, "rb")
    sph = util.read_signal(sph_file, force_as=force_as)
    assert np.all(sph == wav)


@pytest.mark.parametrize(
    "env_var,suffix", [("WSJ_DIR", ".wv1"), ("TIMIT_DIR", ".sph")], ids=["wsj", "timit"]
)
def test_read_sphere_corpus(temp_dir, env_var, suffix):
    num_utts = 50
    env_dir = os.environ.get(env_var, None)
    if env_dir is None:
        pytest.skip(f"Corpus dir not set (export {env_var})")
    sph2pipe_path = os.environ.get("SPH2PIPE", None)
    if sph2pipe_path is None:
        pytest.skip("SPH2PIPE dir not set")
    try:
        import pathlib
    except ImportError:
        pathlib = pytest.importorskip("pathlib2")
    sphere_files = [
        str(x)
        for (x, _) in zip(
            pathlib.Path(env_dir).glob("**/*" + suffix), list(range(num_utts))
        )
    ]
    import subprocess

    wav_files = []
    for utt_idx, src in enumerate(sphere_files):
        wav_file = os.path.join(temp_dir, "{}.wav".format(utt_idx))
        wav_files.append(wav_file)
        assert not subprocess.call([sph2pipe_path, "-f", "wav", src, wav_file])
    for wav_path, sph_path in zip(wav_files, sphere_files):
        wav = util.read_signal(wav_path)
        sph = util.read_signal(sph_path, force_as="sph")
        assert (wav == sph).all()


@pytest.mark.parametrize("key", [True, False], ids=["w/ key", "w/o key"])
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_hdf5(temp_dir, key, from_file):
    h5py = pytest.importorskip("h5py")
    rfilename = os.path.join(temp_dir, "foo.hdf5")
    h5py_file = h5py.File(rfilename, "w")
    h5py_file.create_group("a/b/c")
    h5py_file.create_group("a/b/d/e")
    dset_1 = np.random.random((1000, 2000))
    dset_2 = (np.random.random(10) * 1000).astype(int)
    h5py_file.create_dataset("a/b/d/f", (1000, 2000), data=dset_1)
    h5py_file.create_dataset("g", (10,), data=dset_2)
    h5py_file.close()
    if from_file:
        force_as = None
    else:
        rfilename = open(rfilename, "rb")
        force_as = "hdf5"
    if key:
        dset_3 = util.read_signal(rfilename, key="g", force_as=force_as)
        assert np.allclose(dset_2, dset_3)
    else:
        dset_3 = util.read_signal(rfilename, force_as=force_as)
        assert np.allclose(dset_1, dset_3)


@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_torch(temp_dir, from_file):
    torch = pytest.importorskip("torch")
    torch.manual_seed(10)
    rfilename = os.path.join(temp_dir, "foo.pt")
    exp = torch.randn(10, 4)
    torch.save(exp, rfilename)
    exp = exp.numpy()
    if from_file:
        force_as = None
    else:
        rfilename = open(rfilename, "rb")
        force_as = "pt"
    act = util.read_signal(rfilename, force_as=force_as)
    assert np.allclose(exp, act)


@pytest.mark.parametrize(
    "allow_pickle", [True, False], ids=["picklable", "notpicklable"]
)
@pytest.mark.parametrize("fix_imports", [True, False], ids=["fix", "nofix"])
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_numpy_binary(temp_dir, allow_pickle, fix_imports, from_file):
    rfilename = os.path.join(temp_dir, "foo.npy")
    buff_1 = np.random.random((1000, 10, 5))
    np.save(rfilename, buff_1, allow_pickle=allow_pickle, fix_imports=fix_imports)
    if from_file:
        force_as = None
    else:
        rfilename = open(rfilename, "rb")
        force_as = "npy"
    buff_2 = util.read_signal(rfilename, force_as=force_as)
    assert np.allclose(buff_1, buff_2)


@pytest.mark.parametrize(
    "compressed", [True, False], ids=["compressed", "uncompressed"]
)
@pytest.mark.parametrize("key", [True, False], ids=["withkey", "withoutkey"])
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_numpy_archive(temp_dir, compressed, key, from_file):
    rfilename = os.path.join(temp_dir, "foo.npz")
    buff_1 = np.random.random((5, 1, 2))
    buff_2 = np.random.random((1,))
    if compressed and key:
        np.savez_compressed(rfilename, a=buff_1, b=buff_2)
    elif compressed:
        np.savez_compressed(rfilename, buff_1, buff_2)
    elif key:
        np.savez(rfilename, a=buff_1, b=buff_2)
    else:
        np.savez(rfilename, buff_1, buff_2)
    if from_file:
        force_as = None
    else:
        rfilename = open(rfilename, "rb")
        force_as = "npz"
    if key:
        buff_3 = util.read_signal(rfilename, key="a", force_as=force_as)
    else:
        buff_3 = util.read_signal(rfilename, force_as=force_as)
    assert np.allclose(buff_1, buff_3)


@pytest.mark.parametrize("text", [True, False])
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_numpy_fromfile(temp_dir, text, from_file):
    buff_1 = np.random.random(1000)
    rfilename = os.path.join(temp_dir, "foo")
    sep = "," if text else ""
    buff_1.tofile(rfilename, sep=sep)
    if not from_file:
        rfilename = open(rfilename, mode="r" if text else "rb")
    buff_2 = util.read_signal(rfilename, sep=sep, force_as="file")
    assert np.allclose(buff_1, buff_2)


@pytest.mark.parametrize("filetype", ["wav", "aiff", "flac", "ogg"])
@pytest.mark.parametrize("from_file", [True, False], ids=["file", "buffer"])
def test_read_soundfile(filetype, from_file):
    sf = pytest.importorskip("soundfile")
    if filetype.upper() not in sf.available_formats():
        pytest.skip(f"This version of libsndfile does not support {filetype}")
    filename = os.path.join(os.path.dirname(__file__), "audio", "sin1k." + filetype)
    if from_file:
        buff = util.read_signal(filename, dtype=np.float64)
    else:
        with open(filename, mode="rb") as file_:
            buff = util.read_signal(file_, dtype=np.float64, force_as="soundfile")
    buff *= np.blackman(len(buff))
    pow = np.abs(np.fft.rfft(buff))
    # 8kHz/sec * 1 sec = 8000 samples
    # sin at 1kHz = sample 1000
    assert np.argmax(pow) == 1000


def test_wds_read_signal(temp_dir):
    wds = pytest.importorskip("webdataset")
    N = 5

    buffers_exp = []
    tar_pth = os.path.join(temp_dir, "wds.tar")
    with tarfile.open(tar_pth, "w") as tar:
        for utt_no in range(N):
            buffers_exp_utt = [
                np.random.random((utt_no + 1, utt_no + 2, utt_no + 3)),
                (np.random.random((utt_no + 1) * 1000) * 1000).astype("<i2"),
                {"utt_no": utt_no},
            ]
            paths = [
                os.path.join(temp_dir, f"utt{utt_no}.{x}")
                for x in ("npy", "wav", "json")
            ]
            buffers_exp.append(buffers_exp_utt)
            np.save(paths[0], buffers_exp_utt[0])
            with wave.open(paths[1], "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(buffers_exp_utt[1].tobytes("C"))
            with open(paths[2], "w") as fp:
                json.dump(buffers_exp_utt[2], fp)
            for path in paths:
                tar.add(path)

    ds = (
        wds.WebDataset("file:" + tar_pth.replace("\\", "/"))
        .decode(util.wds_read_signal)
        .to_tuple("npy", "wav", "json")
    )
    buffers_act = list(ds)
    assert len(buffers_act) == N
    for i, buffers_exp_utt, buffers_act_utt in zip(range(N), buffers_exp, buffers_act):
        assert len(buffers_exp_utt) == len(buffers_act_utt) == 3
        buffer_exp, buffer_act = buffers_exp_utt[-1], buffers_act_utt[-1]
        assert buffer_exp == buffer_act, (i, 2)  # json
        for j, buffer_exp, buffer_act in zip(
            range(3), buffers_exp_utt[:-1], buffers_act_utt[:-1]
        ):
            assert buffer_exp.shape == buffer_act.shape, (i, j)
            assert (buffer_exp == buffer_act).all(), (i, j)
