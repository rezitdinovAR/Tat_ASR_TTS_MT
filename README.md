# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

## Installation:

<a name="installation"></a>

**Clone the repo**

```shell
git clone git@github.com:rezitdinovAR/Tat_ASR_TTS_MT.git
cd vits
```

## Setting up the conda env

This is assuming you have navigated to the `vits` root after cloning it.

**NOTE:** This is tested under `python3.11` with conda env. For other python versions, you might encounter version conflicts.
**NOTE:** This is tested under `python3.11` with conda env. For other python versions, you might encounter version conflicts.

**PyTorch 2.0**
Please refer [requirements.txt](requirements.txt)
Please refer [requirements.txt](requirements.txt)

```shell
# install required packages (for pytorch 2.0)
conda create -n vits python=3.11
conda activate vits
pip install -r requirements.txt
```

## Download datasets

1. Custom dataset: You can use your own dataset. Please refer [here](#custom-dataset).

### Custom dataset

1. create a folder with wav files
2. create configuration file in [configs](configs/). Change the following fields in `custom_base.json`:
3. create configuration file in [configs](configs/). Change the following fields in `custom_base.json`:

```js
{
  "data": {
    "training_files": "filelists/custom_audio_text_train_filelist.txt.cleaned", // path to training cleaned filelist
    "validation_files": "filelists/custom_audio_text_val_filelist.txt.cleaned", // path to validation cleaned filelist
    "text_cleaners": ["english_cleaners2"], // text cleaner
    "bits_per_sample": 16, // bit depth of wav files
    "training_files": "filelists/custom_audio_text_train_filelist.txt.cleaned", // path to training cleaned filelist
    "validation_files": "filelists/custom_audio_text_val_filelist.txt.cleaned", // path to validation cleaned filelist
    "text_cleaners": ["english_cleaners2"], // text cleaner
    "bits_per_sample": 16, // bit depth of wav files
    "sampling_rate": 22050, // sampling rate if you resampled your wav files
    ...
    "n_speakers": 0, // number of speakers in your dataset if you use multi-speaker setting
    "cleaned_text": true // if you already cleaned your text (See text_phonemizer.ipynb), set this to true
  },
  ...
    "cleaned_text": true // if you already cleaned your text (See text_phonemizer.ipynb), set this to true
  },
  ...
}
```

3. install espeak-ng (optional)

**NOTE:** This is required for the [preprocess.py](preprocess.py) and [inference.ipynb](inference.py) notebook to work. If you don't need it, you can skip this step. Please refer [espeak-ng](https://github.com/espeak-ng/espeak-ng)

4. preprocess text

You can do this step by step way:

- create a dataset of text files. See [text_dataset.ipynb](text_dataset.ipynb)
- phonemize or just clean up the text. Please refer [text_phonemizer.ipynb](run_phonemes.py)
- create filelists and cleaned version with train test split. See [text_split.ipynb](create_path_map.py)
- rename or create a link to the dataset folder. Please refer [text_split.ipynb](create_path_map.py)

```shell
ln -s /path/to/custom_dataset DUMMY3
```

## Training Examples

```shell
# Custom dataset (multi-speaker)
python train_ms.py -c configs/custom_base.json -m custom_base
```

## Inference Example

See [inference.ipynb](inference.py)
See [inference_batch.ipynb](inference_batch.py) for multiple sentences inference

## Acknowledgements

- This repo is based on [VITS](https://github.com/jaywalnut310/vits)
- Text to phones converter for multiple languages is based on [phonemizer](https://github.com/bootphon/phonemizer)
- We also thank GhatGPT for providing writing assistance.

## References

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
- [A TensorFlow implementation of Google's Tacotron speech synthesis with pre-trained model (unofficial)](https://github.com/keithito/tacotron)
