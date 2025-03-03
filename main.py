# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import librosa
import numpy as np
import sys
import os
import tqdm
import warnings
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import pandas as pd
from diskcache import Cache

from models import whisperx_asr, chunkformer_asr
from utils.tool import (
    export_to_wav,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, silero_vad

warnings.filterwarnings("ignore")
audio_count = 0

cache = Cache("cache")


@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    sample_rate = cfg["entrypoint"]["SAMPLE_RATE"]
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }


@time_logger
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    logger.debug(f"vocals shape before resample: {vocals.shape}")
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"vocals shape after resample: {vocals.shape}")
    audio["waveform"] = vocals[:, 0]  # vocals is stereo, only use one channel

    return audio


# Step 2: Speaker Diarization
@time_logger
def speaker_diarization(audio, num_speakers=None):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        },
        num_speakers=num_speakers
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list, min_length=1.5, max_length=15):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 100  # merge gap in seconds, if smaller than this, merge

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] > max_length:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than {max_length}s, force trimming to {max_length}s smaller segments"
            )
            while segment_end - current_start >= max_length:
                vad["end"] = current_start + max_length  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += max_length
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= min_length
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= max_length
        ):
            updated_list.append(vad)
            continue
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time
            continue
        pass

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= min_length
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


@time_logger
def asr(audio):
    # resample to 16k
    temp_audio = librosa.resample(
        audio["waveform"], orig_sr=audio["sample_rate"], target_sr=16000
    )
    #asr_model.load_asr_model()
    results = asr_model.transcribe_and_align(temp_audio)
    #asr_model.unload_asr_model()
    return results


@time_logger
def batch_asr_dir(audio_dir):
    #asr_model.load_asr_model()
    results = asr_model.transcribe_directory(audio_dir)
    #asr_model.unload_asr_model()
    return results


@time_logger
def mos_prediction(audio, vad_list):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        vad_list (list): List of VAD segments with start and end times.

    Returns:
        tuple: A tuple containing the average MOS and the updated VAD segments with MOS scores.
    """
    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )
    # Load the DNSMOS model
    dnsmos_compute_score.load_model()
    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)

        segment = audio[start:end]

        dnsmos = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos
    dnsmos_compute_score.unload_model()
    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")

    return predict_dnsmos, vad_list


def filter(mos_list, min_duration=1.5, max_duration=15):
    """
    Filter out the segments with MOS scores, wrong char duration, and total duration.

    Args:
        mos_list (list): List of VAD segments with MOS scores.

    Returns:
        list: A list of VAD segments with MOS scores above the average MOS.
    """
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(
        mos_list, min_duration=min_duration, max_duration=max_duration, min_dnsmos=2.0
    )
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    logger.debug(
        f"> {all_segment - filtered_segment}/{all_segment} {(all_segment - filtered_segment) / all_segment:.2%} segments filtered."
    )
    filtered_list = [mos_list[idx] for idx, _ in filtered_audio_stats]
    return filtered_list


@time_logger
def main_process(
    audio_path, save_path=None, audio_name=None, min_length=1.5, max_length=15, num_speakers=None
):
    """
    Process the audio file, including standardization, source separation, speaker segmentation, VAD, ASR, export to MP3, and MOS prediction.

    Args:
        audio_path (str): Audio file path.
        save_path (str, optional): Save path, defaults to None, which means saving in the "_processed" folder in the audio file's directory.
        audio_name (str, optional): Audio file name, defaults to None, which means using the file name from the audio file path.

    Returns:
        tuple: Contains the save path and the MOS list.
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")

    # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path) + "_processed", audio_name
    )
    os.makedirs(save_path, exist_ok=True)
    logger.debug(
        f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
    )

    logger.info(
        "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )
    st_audio = standardization(audio_path)

    if args.separate_speech:
        logger.info("Step 1: Source Separation")
        if debug:
            audio = cache.get(f"separation:{audio_path}", None)
            if audio is None:
                separate_predictor1.load_model()
                audio = source_separation(separate_predictor1, st_audio)
                separate_predictor1.unload_model()
                cache.set(f"separation:{audio_path}", audio)
        else:
            separate_predictor1.load_model()
            audio = source_separation(separate_predictor1, st_audio)
            separate_predictor1.unload_model()
    else:
        logger.info("Step 1: Skipping Source Separation")
        audio = st_audio
        del st_audio

    logger.info("Step 2: Speaker Diarization")
    if debug:
        speakerdia = cache.get(f"diarization:{audio_path}", None)
        if speakerdia is None:
            speakerdia = speaker_diarization(audio, num_speakers)
            cache.set(f"diarization:{audio_path}", speakerdia)
    else:
        speakerdia = speaker_diarization(audio, num_speakers)

    logger.info("Step 3: ASR")
    if debug:
        asrx_result = cache.get(f"asrx_result:{audio_path}", None)
        if asrx_result is None:
            asrx_result = asr(audio)
            cache.set(f"asrx_result:{audio_path}", asrx_result)
    else:
        asrx_result = asr(audio)

    logger.info("Step 4: Fine-grained Segmentation by VAD")
    vad_list = vad.vad(speakerdia, asrx_result, audio)

    segment_list = cut_by_speaker_label(
        vad_list, min_length=min_length, max_length=max_length
    )  # post process after vad

    # asr_model.load_asr_model()
    # asr_result = asr(segment_list, audio)
    # asr_model.unload_asr_model()

    logger.info("Step 5: Filter")
    logger.info("Step 5.1: calculate mos_prediction")
    avg_mos, mos_list = mos_prediction(audio, segment_list)

    logger.info(f"Step 5.1: done, average MOS: {avg_mos}")

    logger.info("Step 5.2: Filter out files with less than average MOS")
    filtered_list = filter(mos_list, min_duration=min_length, max_duration=max_length)

    logger.info("Step 6: write result into WAV and JSON file")
    export_to_wav(audio, filtered_list, save_path, audio_name, 0.15)

    results_directory = batch_asr_dir(save_path)

    for segment in filtered_list:
        if segment.get("path", "NONE") in results_directory:
            segment["original_text"] = segment["text"]
            segment["text"] = (
                results_directory.get(segment.get("path", "NONE"), {})
                .get("segments", [{}])[0]
                .get("text", "")
                .strip()
            )
        else:
            logger.warning(f"ASR result not found for {segment.get('path', 'NONE')}")

    final_path = os.path.join(save_path, audio_name + ".json")
    with open(final_path, "w") as f:
        json.dump(filtered_list, f, ensure_ascii=False)

    metadata = os.path.join(save_path, "metadata.csv")
    with open(metadata, "w") as f:
        f.writelines(
            [
                f"{segment.get('file_name','ERROR')}|{segment.get('text','ERROR')}|{'-'.join(segment.get('speaker',['ERROR']))}\n"
                for segment in filtered_list
            ]
        )

    logger.info(f"All done, Saved to: {final_path}")
    return final_path, filtered_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="large",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--chunkformer_path",
        type="str",
        default="/content/chunkformer-large-vie",
        help="The checkpoint path of ChunkFormer"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=15,
        help="Max duration of generated audio segment in seconds.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.2,
        help="Min duration of generated audio segment in seconds.",
    )
    parser.add_argument(
        "--separate_speech",
        type=bool,
        default=True,
        help="Min duration of generated audio segment in seconds.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="The number of speakers in the dataset, set to 0 to skip speech diarization",
    )
    parser.add_argument(
        "--total_batch_duration",
        type=int,
        default=1800,
        help="The duration of a continious slice batch from audio feeding to ChunkFormer. Higher means better parallelism but also uses more VRAM",
    )
    parser.add_argument(
        "--ctc_max_silence",
        type=int,
        default=10,
        help="Max frames of silece (80ms each) before splitting."
    )
    args = parser.parse_args()
    # Set debug mode for caching expensive operations
    debug = False

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    if args.input_folder_path:
        logger.info(f"Using input folder path: {args.input_folder_path}")
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.debug("Loading models...")

    # Load models
    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name+':0')
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)

    check_env(logger)

    # Speaker Diarization
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf', check the config file. "
            "You can get the token at https://huggingface.co/settings/tokens. "
            "Remeber grant access following https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr"
        )
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=cfg["huggingface_token"],
    )
    dia_pipeline.to(device)

    # ASRX
    logger.debug(" * Loading ASRX Model")
    """asr_model = whisperx_asr.WhisperXModel(
        args.whisper_arch,
        language="vi",
        device="cuda",
        compute_type=args.compute_type,
        threads=args.threads,
        #asr_options={
        #    "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
        #},
    )
    asr_model.load_asr_model()"""
    asr_model = chunkformer_asr.ChunkFormerModel(
        args.chunkformer_path,
        device=device,
        total_batch_duration=args.total_batch_duration,
        max_silence=args.ctc_max_silence
    )
    # VAD
    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(
        device=device, min_duration=args.min_duration, max_duration=args.max_duration
    )

    if args.separate_speech:
        # Background Noise Separation
        logger.debug(" * Loading Background Noise Model")
        separate_predictor1 = separate_fast.Predictor(
            args=cfg["separate"]["step1"], device=device_name
        )
    else:
        logger.debug(" * Skipping Background Noise Separation")

    # DNSMOS Scoring
    logger.debug(" * Loading DNSMOS Model")
    primary_model_path = cfg["mos_model"]["primary_model_path"]
    dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, device_name)
    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

    for path in audio_paths:
        main_process(path, num_speakers=args.num_speakers)
