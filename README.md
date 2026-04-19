# Speech Pipeline

This project implements a modular end-to-end speech processing pipeline for Hinglish lecture audio:

- DeepFilterNet-based denoising with a safe spectral fallback
- Frame-level Hindi vs English LID with a CNN + BiLSTM PyTorch model
- Whisper small ASR with custom beam search and n-gram LM rescoring
- Hinglish-to-IPA G2P using Epitran plus rule-based fallback mappings
- Translation with IndicTrans2 when available, plus dictionary fallback
- Speaker embedding extraction with Resemblyzer
- TTS backends: IndicF5, Indic Parler-TTS, and YourTTS
- Chunked translation + chunked synthesis with chunk concatenation for long-form inputs
- Prosody transfer with F0/energy extraction and memory-safe DTW fallback
- Anti-spoof detection with LFCC + CNN
- FGSM audio attack against the LID model with an SNR floor

## Install

```bash
pip install -e .
```

If you want the exact requested optional backends, also install:

```bash
pip install deepfilternet indictrans2
```

## Run the full pipeline

```bash
python -m speech_pipeline.pipeline \
  --input-audio /path/to/lecture.wav \
  --reference-audio /path/to/60s_reference.wav \
  --output-dir outputs \
  --translation-target hin_Deva \
  --lm-corpus /path/to/corpus.txt \
  --tts-backend parler
```

Supported `--tts-backend` values are: `indicf5`, `parler`, and `yourtts`.

Pipeline outputs now include cleaned/chunked text artifacts in `result.json`:

- `transcript_raw`
- `transcript_chunks`
- `translated_chunks`
- `num_tts_chunks`

## Resume from Existing TTS chunks

If a long TTS run is interrupted, you can skip ASR/translation/new TTS generation and continue from already created chunk WAVs (`output_dir/tts_chunks/*.wav`):

```bash
python -m speech_pipeline.pipeline \
  --input-audio /path/to/lecture.wav \
  --reference-audio /path/to/60s_reference.wav \
  --output-dir outputs \
  --clean-audio-path outputs/clean.wav \
  --tts-backend parler \
  --combine-existing-tts-chunks-only
```

## Build the ASR corpus

If you have both Hindi and English subtitle files for the lecture, build the LM corpus first and pass the resulting text file to `--lm-corpus`:

```bash
python -m speech_pipeline.asr.build_corpus \
  --input raw_data/lecture.en.srt raw_data/lecture.hi.srt \
  --output corpus.txt
```

For a mixed Hinglish lecture, using both subtitle files together is usually the best choice.

## Train the LID model

```bash
python -m speech_pipeline.lid.train \
  --manifest lid_manifest.csv \
  --output lid_checkpoint.pt
```

## Preprocess a long audio file for LID training

```bash
python -m speech_pipeline.lid.preprocess \
  --audio /path/to/long_video_audio.wav \
  --output-dir lid_prep \
  --whisper-model-path models/whisper-small \
  --chunk-seconds 30 \
  --overlap-seconds 0 \
  --exclude-start-sec 8400 \
  --exclude-end-sec 9240
```

This will:

- split the long WAV into chunk files under `lid_prep/chunks/`
- save frame-label arrays under `lid_prep/frame_labels/`
- write a training manifest at `lid_prep/lid_manifest.csv`

If you only want a specific region, you can use `--start-sec` and `--end-sec` instead of the exclusion flags.

## Prepare a 10-minute LID evaluation segment

This extracts a held-out 10-minute slice from the source audio and preprocesses it into LID-ready chunks.

```bash
python -m speech_pipeline.lid.prepare_eval_segment \
  --source-audio /path/to/full_video_audio.wav \
  --output-dir lid_eval \
  --segment-start-sec 8400 \
  --segment-duration-sec 600 \
  --whisper-model-path models/whisper-small
```

This will:

- save the extracted segment as `lid_eval/eval_segment.wav`
- preprocess that segment into `lid_eval/lid_prep/`
- write a manifest at `lid_eval/lid_prep/lid_manifest.csv`

## Evaluate the LID model

Use a manifest with `audio_path` and `frame_labels` for the evaluation segment.

```bash
python -m speech_pipeline.lid.evaluate \
  --manifest lid_eval/lid_prep/lid_manifest.csv \
  --checkpoint checkpoints/lid_model.pt \
  --device cuda \
  --output lid_eval_report.json
```

This reports:

- frame-level accuracy
- `precision`, `recall`, and `F1` for English and Hindi
- macro-F1 across both languages
- language-switch precision, recall, F1, and timing error

For a meaningful benchmark, the `frame_labels` for the evaluation segment should ideally be manually verified. The provided preprocessing step can generate pseudo-labels, but those are best treated as a bootstrap, not final ground truth.

## Train the anti-spoof model

For long files, create chunk-based train/eval manifests first:

```bash
python -m speech_pipeline.antispoof.prepare_manifest \
  --bona-fide-audio /path/to/real_1.wav /path/to/real_2.wav \
  --spoof-audio /path/to/tts_1.wav /path/to/tts_2.wav \
  --output-dir outputs/antispoof \
  --chunk-seconds 8 \
  --chunk-overlap-seconds 1 \
  --train-ratio 0.8
```

Then train:

```bash
python -m speech_pipeline.antispoof.train \
  --manifest outputs/antispoof/antispoof_train_manifest.csv \
  --output checkpoints/antispoof_model.pt \
  --device cuda \
  --chunk-seconds 8 \
  --chunk-overlap-seconds 1
```

## Evaluate the system

```bash
python -m speech_pipeline.evaluate \
  --asr-manifest eval_asr.csv \
  --lid-manifest eval_lid.csv \
  --mcd-manifest eval_mcd.csv \
  --spoof-manifest eval_spoof.csv \
  --attack-manifest eval_attack.csv \
  --lid-checkpoint lid_checkpoint.pt \
  --antispoof-checkpoint antispoof_checkpoint.pt \
  --lm-corpus corpus.txt \
  --spoof-threshold 0.5 \
  --output evaluation_report.json
```

Only the metrics whose manifests are provided are executed. For example, passing only `--attack-manifest` runs only the adversarial robustness evaluation.

Expected evaluation manifests:

- `eval_asr.csv`: `audio_path,reference_text,language`
- `eval_lid.csv`: `audio_path,switch_times`
- `eval_mcd.csv`: `reference_audio,synthesized_audio`
- `eval_spoof.csv`: `audio_path,label` (optional: `start_sec,end_sec` for chunk-level evaluation)
- `eval_attack.csv`: `audio_path`

Useful spoof-eval options:

- `--spoof-chunk-seconds`
- `--spoof-chunk-overlap-seconds`
- `--spoof-max-chunks`
- `--spoof-threshold`

## Manifest format

For training, each manifest CSV should include at least:

- `audio_path`
- `label`

The LID manifest may also include `frame_labels` as a `.npy` path or a semicolon-separated label sequence.

## Example CSV rows

### LID training

```csv
audio_path,label,frame_labels
data/lid/utt_001.wav,0,data/lid/utt_001_frames.npy
data/lid/utt_002.wav,1,data/lid/utt_002_frames.npy
```

If you do not have frame labels yet, you can start with clip-level labels only:

```csv
audio_path,label
data/lid/utt_003.wav,0
data/lid/utt_004.wav,1
```

### Anti-spoof training

```csv
audio_path,label
data/spoof/real_001.wav,0
data/spoof/tts_001.wav,1
```

Chunk-level anti-spoof manifests may include explicit segment bounds:

```csv
audio_path,label,start_sec,end_sec
data/spoof/real_long.wav,0,0.0,8.0
data/spoof/tts_long.wav,1,0.0,8.0
```

### ASR evaluation

```csv
audio_path,reference_text,language
data/eval/asr_001.wav,"we are studying stochastic processes today",en
data/eval/asr_002.wav,"ye lecture bahut important hai",hi
```

### LID switch evaluation

`switch_times` can be a JSON list or a delimiter-separated list of seconds.

```csv
audio_path,switch_times
data/eval/lid_001.wav,"[0.62, 1.48, 2.10]"
data/eval/lid_002.wav,"0.55;1.33;2.40"
```

### MCD evaluation

```csv
reference_audio,synthesized_audio
data/mcd/ref_001.wav,data/mcd/synth_001.wav
data/mcd/ref_002.wav,data/mcd/synth_002.wav
```

### Attack evaluation

```csv
audio_path
data/attack/utt_001.wav
data/attack/utt_002.wav
```

Attack report fields:

- `min_epsilon_mean`
- `min_epsilon_median`
- `snr_db_mean`
- `num_samples`
