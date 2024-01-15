#%matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
import librosa
import soundfile as sf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
import os
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

os.environ['CUDA_VISIBLE_DEVICES']='7'

hps = utils.get_hparams_from_file("./configs/custom_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/custom_base/G_201000.pth", net_g, None)

text = "Татарстан Рәисе шулай ук Төркия Республикасының Татарстанның әйдәп баручы чит ил партнерларының берсе булуын билгеләп үтте. Төркия Генераль Консуллыгының Татарстан Республикасы Дәүләт органнары белән хезмәттәшлеге югары дәрәҗәдә бара."

stn_tst = get_text(text, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
sf.write('out/'+text[:5]+'_'+check_point[check_point.rfind('/')+1:check_point.rfind('.')]+'.wav', audio, hps.data.sampling_rate)