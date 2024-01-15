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

hps = utils.get_hparams_from_file("./configs/multispeaker.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

check_point = "./logs/multispeaker/G_80000.pth"

_ = utils.load_checkpoint(check_point, net_g, None)

text ="Татарстан Рәисе шулай ук Төркия Республикасының Татарстанның әйдәп баручы чит ил партнерларының берсе булуын билгеләп үтте. Төркия Генераль Консуллыгының Татарстан Республикасы Дәүләт органнары белән хезмәттәшлеге югары дәрәҗәдә бара."


"Бүген Татарстан Республикасы Рәисе Рөстәм Миңнеханов Анкара шәһәре мэры Мансур Яваш белән очрашты. Очрашу Казан Кремлендә Рөстәм Миңнехановның эш кабинетында узды. Бу чаралар ихтыяҗ булуны исбатлады һәм ел ахырына кадәр республикада үткәреләчәк, дип төгәлләде ул. Мәсәлән, 12 ярминкә көнендә 930 млн сумлык авыл хуҗалыгы продукциясе сатылган.И Ходаем, оныкларыма, нәсел-нәсәбемә мондый хәлләрне күрсәтмә. Әнә бер бала ничек илереп елый: тибенә, нәрсәдер таләп итә. Күз яшьләре тәгәрәп мендәргә төшә. Бу мендәрләр, бу юрганнар күпме күз яшьләрен үзенә сеңдергәннәр икән?! Ә елау тавышлары? Ул тавышларны таш стеналар, бетон түшәмнәр генә йотып бетерә алмый."



'Онлайн-кинотеатрларда "Егет сүзе" сериалы күрсәтелә башлады. 80-90нчы елларда Казан яшьләренең криминаль төркемнәргә туплану тарихын тасвирлаган бу киноны Казанда төшерергә рөхсәт итмәгәннәр. Бу хәлләр бүген дә кабатлана аламы?'

"Алсу ничек хәлләр? Син буген университетта булдыңмы?"

"Татарстан Рәисе Рөстәм Миңнеханов Мәскәү өлкәсенә эш сәфәре кысаларында «Гринвуд» бизнес-паркында булды һәм Кытайның эшлекле даирәләре вәкилләре белән очрашты. «Гринвуд» бизнес-паркы 2010 елда Кытай капиталы катнашында төзелгән. Россия һәм КХР арасында икътисадый һәм гуманитар хезмәттәшлек өчен платформа булып тора. Инфраструктура үз эченә кунакханә, конгресс-үзәк, сөйләшүләр бүләмәләре, банк бүлекчәләре һәм башка күп нәрсәләрне ала. Бизнес-парк территориясендә 12 мең хезмәткәр эшли. Мәйданчыкта 300дән артык төрле компаниянең штаб-фатирлары һәм офислары тәкъдим ителгән, алар арасында 170 Кытай компаниясе. Бизнес-парк резидентларының Россия һәм Кытай арасындагы гомуми товар әйләнешенә керткән гомуми өлеше 6 млрд доллар тәшкил итә."

stn_tst = get_text(text, hps)
print('here')
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([0]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
sf.write('out/'+text[:5]+'_'+check_point[check_point.rfind('/')+1:check_point.rfind('.')]+'.wav', audio, hps.data.sampling_rate)
