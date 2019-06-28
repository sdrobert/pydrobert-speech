from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import wave

import numpy as np
import pytest

from pydrobert.speech import command_line


def test_compute_feats_from_kaldi_tables(temp_dir):
    kaldi_io = pytest.importorskip('pydrobert.kaldi.io')
    np.random.seed(5)
    wav_scp = os.path.join(temp_dir, 'wav.scp')
    feat_ark = os.path.join(temp_dir, 'feat.ark')
    wav_file_dir = os.path.join(temp_dir, 'wav')
    num_utts = 100
    max_samples = 32000
    sampling_rate = 16000
    num_digits = int(np.log10(num_utts)) + 1
    utt_fmt = 'utt{{:0{}d}}'.format(num_digits)
    computer_json_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data',
        'fbank.json'
    )
    preprocessor_json_path = os.path.join(temp_dir, "preprocess.json")
    postprocessor_json_path = os.path.join(temp_dir, "unit.json")
    with open(preprocessor_json_path, 'w') as f:
        f.write('["dither"]\n')
    with open(postprocessor_json_path, 'w') as f:
        f.write('["unit"]\n')
    if not os.path.isdir(wav_file_dir):
        os.makedirs(wav_file_dir)
    with open(wav_scp, 'w') as scp:
        for utt_idx in range(num_utts):
            utt_id = utt_fmt.format(utt_idx)
            wav_path = os.path.join(wav_file_dir, utt_id + '.wav')
            num_samples = np.random.randint(max_samples)
            signal = np.random.randint(
                -2 ** 15, 2 ** 15 - 1, num_samples, dtype=np.int16)
            wv = wave.open(wav_path, 'wb')
            wv.setnchannels(1)
            wv.setsampwidth(2)
            wv.setframerate(16000)
            wv.writeframes(signal.tobytes())
            wv.close()
            scp.write('{} {}\n'.format(utt_id, wav_path))
    args = [
        'scp,s:' + wav_scp, 'ark:' + feat_ark, computer_json_path,
        "--seed=30", "--preprocess={}".format(preprocessor_json_path),
        '--postprocess={}'.format(postprocessor_json_path),
    ]
    assert not command_line.compute_feats_from_kaldi_tables(args)
    exps = []
    with kaldi_io.open('ark,s:' + feat_ark, 'bm') as feats:
        not_all_feats = 0
        for utt_idx, feat in enumerate(feats):
            not_all_feats |= feat.shape[0]
            assert feat.shape[0] == 0 or feat.shape[-1] == 40
            exps.append(feat)
        assert utt_idx == num_utts - 1
        assert not_all_feats
    np.random.seed(300)
    assert not command_line.compute_feats_from_kaldi_tables(args)
    with kaldi_io.open('ark,s:' + feat_ark, 'bm') as feats:
        for utt_idx, (exp, act) in enumerate(zip(exps, feats)):
            assert np.allclose(act, exp)


def test_signals_to_torch_feat_dir(temp_dir):
    torch = pytest.importorskip('torch')
    torch.manual_seed(50)
    feat_dir = os.path.join(temp_dir, 'feat')
    raw_dir = os.path.join(temp_dir, 'raw')
    map_path = os.path.join(temp_dir, 'map')
    computer_json_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data',
        'fbank.json'
    )
    preprocessor_json_path = os.path.join(temp_dir, "preprocess.json")
    with open(preprocessor_json_path, 'w') as f:
        f.write('["dither"]\n')
    assert os.path.isfile(computer_json_path)
    num_utts, max_samples = 100, 1600
    num_digits = torch.log10(torch.tensor(float(num_utts))).long().item() + 1
    utt_fmt = 'utt{{:0{}d}}'.format(num_digits)
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)
    utt_ids = []
    utt2signal = dict()
    with open(map_path, 'w') as mp:
        for utt_idx in range(num_utts):
            utt_id = utt_fmt.format(utt_idx)
            utt_ids.append(utt_id)
            num_samples = torch.randint(1, max_samples, (1,)).long().item()
            signal = torch.randint(
                -(2 ** 15), 2 ** 15 - 1, (num_samples,), dtype=torch.float32)
            utt2signal[utt_id] = signal
            file_type = torch.randint(3, (1,)).long().item()
            if file_type == 2:  # wave file
                path = os.path.join(raw_dir, '{}.wav'.format(utt_idx))
                wv = wave.open(path, 'wb')
                wv.setnchannels(1)
                wv.setsampwidth(2)
                wv.setframerate(16000)
                wv.writeframes(signal.to(torch.int16).numpy().tobytes())
                wv.close()  # py2.7 doesn't have context manager
            elif file_type == 1:   # npy file
                path = os.path.join(raw_dir, '{}.npy'.format(utt_idx))
                np.save(path, signal.numpy())
            else:   # torch file
                path = os.path.join(raw_dir, '{}.pt'.format(utt_idx))
                torch.save(signal, path)
            mp.write('{} {}\n'.format(utt_id, path))
    args = [map_path, computer_json_path, feat_dir]
    assert not command_line.signals_to_torch_feat_dir(args)
    for utt_id in utt_ids:
        feat = torch.load(os.path.join(feat_dir, '{}.pt'.format(utt_id)))
        assert feat.shape[-1] == 40
        del feat
    args.pop(1)  # remove computer -- storing raw
    torch.manual_seed(30)
    assert not command_line.signals_to_torch_feat_dir(args)
    for utt_id, exp in utt2signal.items():
        act = torch.load(os.path.join(feat_dir, '{}.pt'.format(utt_id)))
        assert torch.allclose(exp, act.flatten())
    torch.manual_seed(40)
    args.append("--seed=1")
    args.append("--preprocess={}".format(preprocessor_json_path))
    assert not command_line.signals_to_torch_feat_dir(args)
    for utt_id in utt_ids:
        signal = utt2signal[utt_id]
        noisy = torch.load(os.path.join(feat_dir, '{}.pt'.format(utt_id)))
        assert not torch.allclose(signal, noisy.flatten())
        utt2signal[utt_id] = noisy
    torch.manual_seed(70)
    args.append('--num-workers=4')
    assert not command_line.signals_to_torch_feat_dir(args)
    for utt_id, exp in utt2signal.items():
        act = torch.load(os.path.join(feat_dir, '{}.pt'.format(utt_id)))
        assert torch.allclose(exp, act)
