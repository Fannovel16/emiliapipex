# Source: https://github.com/snakers4/silero-vad
#
# Copyright (c) 2024 snakers4
#
# This code is from a MIT-licensed repository. The full license text is available at the root of the source repository.
#
# Note: This code has been modified to fit the context of this repository.

import librosa
import torch
import numpy as np

SAMPLING_RATE = 16000


class SileroVAD:
    """
    Voice Activity Detection (VAD) using Silero-VAD.
    """

    def __init__(
        self,
        local=False,
        model="silero_vad",
        device=torch.device("cpu"),
        min_duration=1.5,
        max_duration=15,
    ):
        """
        Initialize the VAD object.

        Args:
            local (bool, optional): Whether to load the model locally. Defaults to False.
            model (str, optional): The VAD model name to load. Defaults to "silero_vad".
            device (torch.device, optional): The device to run the model on. Defaults to 'cpu'.

        Returns:
            None

        Raises:
            RuntimeError: If loading the model fails.
        """
        self.max_duration = max_duration
        self.min_duration = min_duration
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad" if not local else "vad/silero-vad",
                model=model,
                force_reload=False,
                onnx=True,
                source="github" if not local else "local",
            )
            self.vad_model = vad_model
            (get_speech_timestamps, _, _, _, _) = utils
            self.get_speech_timestamps = get_speech_timestamps
        except Exception as e:
            raise RuntimeError(f"Failed to load VAD model: {e}")

    def segment_speech(
        self, audio_segment, start_time, end_time, sampling_rate, max_duration=15
    ):
        """
        Segment speech from an audio segment and return a list of timestamps.

        Args:
            audio_segment (np.ndarray): The audio segment to be segmented.
            start_time (int): The start time of the audio segment in frames.
            end_time (int): The end time of the audio segment in frames.
            sampling_rate (int): The sampling rate of the audio segment.
            max_duration (int): The maximum duration of a segment in seconds.

        Returns:
            list: A list of timestamps, each containing the start and end times of speech segments in frames.

        Raises:
            ValueError: If the audio segment is invalid.
        """
        if audio_segment is None or not isinstance(audio_segment, (np.ndarray, list)):
            raise ValueError("Invalid audio segment")

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=sampling_rate
        )

        adjusted_timestamps = [
            (ts["start"] + start_time, ts["end"] + start_time)
            for ts in speech_timestamps
        ]
        if not adjusted_timestamps:
            return []

        intervals = [
            end[0] - start[1]
            for start, end in zip(adjusted_timestamps[:-1], adjusted_timestamps[1:])
        ]

        segments = []
        max_duration_frames = max_duration * sampling_rate

        def split_timestamps(start_index, end_index):
            segment_start_time = adjusted_timestamps[start_index][0]
            segment_end_time = adjusted_timestamps[end_index][1]
            segment_duration = segment_end_time - segment_start_time

            if start_index == end_index or segment_duration < max_duration_frames:
                segments.append([start_index, end_index])
            else:
                if not intervals[start_index:end_index]:
                    return
                max_interval_index = intervals[start_index:end_index].index(
                    max(intervals[start_index:end_index])
                )
                split_index = start_index + max_interval_index
                split_timestamps(start_index, split_index)
                split_timestamps(split_index + 1, end_index)

        split_timestamps(0, len(adjusted_timestamps) - 1)

        # Further split segments that are longer than max_duration
        final_segments = []
        for start, end in segments:
            segment_start_time = adjusted_timestamps[start][0]
            segment_end_time = adjusted_timestamps[end][1]
            segment_duration = segment_end_time - segment_start_time

            if segment_duration > max_duration_frames:
                current_start = segment_start_time
                while current_start < segment_end_time:
                    current_end = min(
                        current_start + max_duration_frames, segment_end_time
                    )
                    final_segments.append((current_start, current_end))
                    current_start = current_end
            else:
                final_segments.append((segment_start_time, segment_end_time))

        return final_segments

    def vad(self, speakerdia, whisperx_results, audio):
        """
        Process the audio based on the given speaker diarization dataframe.

        Args:
            speakerdia (pd.DataFrame): The diarization dataframe containing start, end, and speaker info.
            whisperx_results (dict): The WhisperX results dictionary containing transcribed segments and timestamps.
            audio (dict): A dictionary containing the audio waveform and sample rate.

        Returns:
            list: A list of dictionaries containing processed audio segments with start, end, and speaker.
        """
        sampling_rate = audio["sample_rate"]
        audio_data = audio["waveform"]
        dia_index = 0
        results_index = 0
        out = []
        last_end = 0
        last_start = -1
        speakers_seen = set()
        count_id = 0
        last_text = []
        last_words = []
        speakers = set()

        for index, segment in enumerate(whisperx_results.get("segments")):
            text = segment["text"].strip()
            words = segment["words"]
            # Join text if it ends with a comma or
            if text.strip().endswith(",") or (
                text.strip()[-1] and text.strip()[-1].isalnum()
            ):
                last_start = float(segment["start"])
                last_text.append(text.strip())
                last_words.extend(segment["words"])
                continue
            if len(last_text) > 0:
                x_start = last_start
                text = " ".join(last_text + [text.strip()])
                last_text = []
                words = last_words + words
                last_words = []
            else:
                x_start = float(segment["start"])
            x_end = float(segment["end"])
            if x_end - x_start < self.min_duration:
                last_start = x_start
                last_text.append(text.strip())
                last_words.extend(segment["words"])
                continue
            temp_dia_index = dia_index
            dia_index_start = 0
            while temp_dia_index < len(speakerdia):
                if temp_dia_index <= 0:
                    previous_dia = 100000000
                else:
                    previous_dia = abs(
                        speakerdia.loc[temp_dia_index - 1, "start"] - x_start
                    )
                current_dia = abs(speakerdia.loc[temp_dia_index, "start"] - x_start)
                if temp_dia_index >= len(speakerdia) - 1:
                    next_dia = 100000000
                else:
                    next_dia = abs(
                        speakerdia.loc[temp_dia_index + 1, "start"] - x_start
                    )
                if previous_dia >= current_dia <= next_dia:
                    dia_index_start = temp_dia_index
                    break
                temp_dia_index += 1
            temp_dia_index = dia_index
            dia_index_end = 0
            while temp_dia_index < len(speakerdia):
                if temp_dia_index == 0:
                    previous_dia = 100000000
                else:
                    previous_dia = abs(
                        speakerdia.loc[temp_dia_index - 1, "end"] - x_end
                    )
                current_dia = abs(speakerdia.loc[temp_dia_index, "end"] - x_end)
                if temp_dia_index == len(speakerdia) - 1:
                    next_dia = 1000000000
                else:
                    # make sure the next segments ends are not overlapping
                    if (
                        speakerdia.loc[temp_dia_index + 1, "end"]
                        < speakerdia.loc[temp_dia_index, "end"]
                    ):
                        index_offset = 2
                        while temp_dia_index + index_offset <= len(speakerdia):
                            if (
                                speakerdia.loc[temp_dia_index + index_offset, "end"]
                                > speakerdia.loc[temp_dia_index, "end"]
                            ):
                                break
                            index_offset += 1
                        if temp_dia_index + index_offset >= len(speakerdia):
                            next_dia = 1000000000
                        else:
                            next_dia = abs(
                                speakerdia.loc[temp_dia_index + index_offset, "end"]
                                - x_end
                            )
                    else:
                        next_dia = abs(
                            speakerdia.loc[temp_dia_index + 1, "end"] - x_end
                        )
                if previous_dia >= current_dia <= next_dia:
                    dia_index_end = temp_dia_index
                    break
                temp_dia_index += 1
            for x in range(dia_index_start, dia_index_end + 1):
                speakers.add(speakerdia.loc[x, "speaker"])
                speakers_seen.add(speakerdia.loc[x, "speaker"])

            dia_index = dia_index_end

            start = speakerdia.loc[dia_index_start, "start"]
            end = speakerdia.loc[dia_index_end, "end"]
            if len(out) == 0:
                left_boundary = 0
            else:
                left_boundary = out[-1]["end"]
            if index == len(whisperx_results.get("segments")) - 1:
                right_boundary = len(audio_data) / sampling_rate
            else:
                right_boundary = float(
                    whisperx_results.get("segments")[index + 1]["start"]
                )

            alt_start, alt_end = self.more_vad(
                audio_data, sampling_rate, left_boundary, start, end, right_boundary
            )

            if end - start <= self.max_duration:
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": alt_start,  # in seconds
                        "end": alt_end,
                        "text": text,
                        "speaker": list(speakers),  # same for all
                        "original_start": x_start,
                        "original_end": x_end,
                        "original_vad_start": start,
                        "original_vad_end": end,
                    }
                )
                speakers = set()
                count_id += 1
                continue

            count_id, temp_out = self.segment_sample(
                audio_data,
                count_id,
                end,
                sampling_rate,
                speakers,
                start,
                text,
                x_end,
                x_start,
                words,
                previous_end=left_boundary,
                next_start=right_boundary,
            )
            out.extend(temp_out)
        return out

    def more_vad(
        self,
        audio_data,
        sampling_rate,
        left_boundary,
        start,
        end,
        right_boundary,
        window=0.25,
    ):
        (
            left_start,
            right_start,
            left_end,
            right_end,
            left_start_2,
            right_end_2,
        ) = self.calculate_local_minimums_within_bounds(
            audio_data,
            sampling_rate,
            left_boundary,
            start,
            end,
            right_boundary,
            window=window,
        )

        # get dB for each possible start and end
        start_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=left_start,
            window_size_s=0.03,
        )
        end_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=right_end,
            window_size_s=0.03,
        )
        alt_start_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=left_start_2,
            window_size_s=0.03,
        )
        alt_end_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=right_end_2,
            window_size_s=0.03,
        )
        og_start_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=start,
            window_size_s=0.03,
        )
        og_end_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=end,
            window_size_s=0.03,
        )
        reverse_start_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=right_start,
            window_size_s=0.03,
        )
        reverse_end_db = self.calculate_db(
            audio_segment=audio_data,
            sample_rate=sampling_rate,
            timestamp_s=left_end,
            window_size_s=0.03,
        )
        start_options = [
            (start, og_start_db),
            (left_start, start_db),
            (right_start, reverse_start_db),
            (left_start_2, alt_start_db),
        ]
        end_options = [
            (end, og_end_db),
            (right_end, end_db),
            (left_end, reverse_end_db),
            (right_end_2, alt_end_db),
        ]

        # sort by db
        start_options.sort(key=lambda x: x[1])
        end_options.sort(key=lambda x: x[1])

        # return the lowest dB start and end
        return start_options[0][0], end_options[0][0]

    def segment_sample(
        self,
        audio_data,
        count_id,
        end,
        sampling_rate,
        speakers,
        start,
        text,
        x_end,
        x_start,
        words,
        previous_end,
        next_start,
    ):
        """
        Split a segment that is too long into smaller segments based on vad.
        """
        temp_out = []
        start_frame = int(start * sampling_rate)
        end_frame = int(end * sampling_rate)
        temp_audio = audio_data[start_frame:end_frame]
        # resample from 24k to 16k
        temp_audio_resampled = librosa.resample(
            temp_audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE
        )
        frames = self.segment_speech(
            temp_audio_resampled,
            int(start * SAMPLING_RATE),
            int(end * SAMPLING_RATE),
            SAMPLING_RATE,
            max_duration=self.max_duration,
        )
        for index, frame in enumerate(frames):
            start_frame_sub = frame[0]
            end_frame_sub = frame[1]
            start_time = start_frame_sub / SAMPLING_RATE
            end_time = end_frame_sub / SAMPLING_RATE
            times = []
            start_best = 1000000
            start_candidate = 0
            end_best = 1000000
            end_candidate = 0
            text = ""

            for word_index, word in enumerate(words):
                if "start" not in word or "end" not in word:
                    continue
                if abs(start_time - word["start"]) < start_best:
                    start_best = abs(start_time - word["start"])
                    start_candidate = word_index
                if abs(end_time - word["end"]) < end_best:
                    end_best = abs(end_time - word["end"])
                    end_candidate = word_index
            text = " ".join(
                [
                    word.get("word", "")
                    for word in words[start_candidate : end_candidate + 1]
                ]
            )

            if len(temp_out) == 0:
                left_boundary = previous_end
            else:
                left_boundary = temp_out[-1]["end"]
            if index == len(frames) - 1:
                right_boundary = next_start
            else:
                right_boundary = frames[index + 1][0] / SAMPLING_RATE

            alt_start, alt_end = self.more_vad(
                audio_data=audio_data,
                sampling_rate=sampling_rate,
                left_boundary=left_boundary,
                start=start_time,
                end=end_time,
                right_boundary=right_boundary,
                window=0.5,
            )

            temp_out.append(
                {
                    "index": str(count_id).zfill(5),
                    "start": alt_start,  # in seconds
                    "end": alt_end,
                    "text": text,
                    "speaker": list(speakers),
                    "original_start": x_start,
                    "original_end": x_end,
                    "original_vad_start": start_time,
                    "original_vad_end": end_time,
                    "split": True,
                }
            )
            count_id += 1
        return count_id, temp_out

    def calculate_local_minimums_within_bounds(
        self,
        audio_segment,
        sampling_rate,
        left_boundary,
        start,
        end,
        right_boundary,
        window=0.5,
    ) -> tuple[float, ...]:
        left_window = min(start - left_boundary, window)
        right_window = min(right_boundary - end, window)

        # plot_rms_trend(audio_segment, end-200, end+200, 1)
        left_start = self.find_local_minimum(
            audio_segment=audio_segment,
            sampling_rate=sampling_rate,
            given_time_s=start,
            window=left_window,
            direction="left",
            smoothing_window_size=3,
        )
        left_start_2 = self.find_local_minimum(
            audio_segment=audio_segment,
            sampling_rate=sampling_rate,
            given_time_s=max(left_boundary + 0.001, left_start - 0.001),
            window=left_window,
            direction="left",
            smoothing_window_size=3,
            include_current=False,
        )
        right_start = self.find_local_minimum(
            audio_segment=audio_segment,
            sampling_rate=sampling_rate,
            given_time_s=start,
            window=min(window, end - start - 0.001),
            direction="right",
            smoothing_window_size=3,
        )
        left_end = self.find_local_minimum(
            audio_segment=audio_segment,
            sampling_rate=sampling_rate,
            given_time_s=end,
            window=min(window, end - start - 0.001),
            direction="left",
            smoothing_window_size=3,
        )
        right_end = self.find_local_minimum(
            audio_segment=audio_segment,
            sampling_rate=sampling_rate,
            given_time_s=end,
            window=right_window,
            direction="right",
            smoothing_window_size=3,
        )
        right_end_2 = self.find_local_minimum(
            audio_segment=audio_segment,
            sampling_rate=sampling_rate,
            given_time_s=min(right_end + 0.001, right_boundary - 0.001),
            window=right_window,
            direction="right",
            smoothing_window_size=3,
            include_current=False,
        )
        return tuple(
            [left_start, right_start, left_end, right_end, left_start_2, right_end_2]
        )

    @staticmethod
    def find_local_minimum(
        *,
        audio_segment,
        sampling_rate,
        given_time_s,
        window,
        direction="right",
        smoothing_window_size=3,
        include_current=True,
    ) -> float:
        """
        Finds the time of a local minimum RMS energy around a given time point in an audio segment.

        Parameters:
        - audio_segment: The complete AudioSegment to analyze.
        - given_time_s: The given time in seconds around which to find the local minimum.
        - window: The time window in milliseconds to search within, to the left or right.
        - direction: The direction to search for the local minimum ('left' or 'right').
        - smoothing_window_size: The size of the moving average window for smoothing.

        Returns:
        - The time in milliseconds of the local minimum RMS energy near the given time.
        """
        given_frame = int(given_time_s * sampling_rate)
        window_frame = int(window * sampling_rate)
        # Extract the audio segment based on the direction
        if direction == "right":
            segment_to_analyze = audio_segment[given_frame : given_frame + window_frame]
        elif direction == "left":
            segment_to_analyze = audio_segment[
                max(0, given_frame - window_frame) : given_frame
            ]
        else:
            raise ValueError("Direction must be 'left' or 'right'.")

        # # Export the AudioSegment to a buffer in WAV format
        # buffer = io.BytesIO()
        # segment_to_analyze.export(buffer, format="wav")
        # buffer.seek(0)
        #
        # # Load the audio segment using Librosa
        # y, sr = librosa.load(buffer, sr=None)

        # Compute the RMS energy
        rms_energy = librosa.feature.rms(y=segment_to_analyze)[0]

        # Apply a moving average to smooth the RMS energy
        smoothed_rms = np.convolve(
            rms_energy,
            np.ones(smoothing_window_size) / smoothing_window_size,
            mode="valid",
        )

        # Find the index of the local minimum in the smoothed RMS energy
        local_min_index = None
        if direction == "right":
            if (
                len(smoothed_rms) > 2
                and include_current
                and smoothed_rms[0] < smoothed_rms[1]
                and smoothed_rms[0] < smoothed_rms[2]
            ):
                local_min_index = 0
            else:
                for i in range(1, len(smoothed_rms) - 1):
                    if (
                        smoothed_rms[i] < smoothed_rms[i - 1]
                        and smoothed_rms[i] < smoothed_rms[i + 1]
                    ) or smoothed_rms[i] == 0:
                        local_min_index = i
                        break
        elif direction == "left":
            reversed_smoothed_rms = smoothed_rms[::-1]
            if (
                len(reversed_smoothed_rms) > 2
                and include_current
                and reversed_smoothed_rms[0] < reversed_smoothed_rms[1]
                and reversed_smoothed_rms[0] < reversed_smoothed_rms[2]
            ):
                local_min_index = len(smoothed_rms)
            else:
                for i in range(1, len(reversed_smoothed_rms) - 1):
                    if (
                        reversed_smoothed_rms[i] < reversed_smoothed_rms[i - 1]
                        and reversed_smoothed_rms[i] < reversed_smoothed_rms[i + 1]
                    ) or reversed_smoothed_rms[i] == 0:
                        local_min_index = len(smoothed_rms) - 1 - i
                        break

        # Handle case where no local minimum is found
        if local_min_index is None:
            return given_time_s

        # Calculate the corresponding time in seconds of the local minimum
        local_min_time_s = librosa.frames_to_time(
            local_min_index + smoothing_window_size // 2, sr=sampling_rate
        )

        # Adjust the time based on the direction and convert to milliseconds
        if direction == "right":
            local_min_time_s = local_min_time_s + given_time_s
        elif direction == "left":
            local_min_time_s = max(0.0, given_time_s - (window - local_min_time_s))
        else:
            raise ValueError("Direction must be 'left' or 'right'.")
        return local_min_time_s

    @staticmethod
    def calculate_db(*, audio_segment, sample_rate, timestamp_s, window_size_s=0.03):
        # Calculate start and end times in seconds
        start_s = max(
            0, timestamp_s - window_size_s / 2
        )  # Ensure start_ms is not negative
        end_s = timestamp_s + window_size_s / 2

        start_frame = int(start_s * sample_rate)
        end_frame = int(end_s * sample_rate)

        # Extract the relevant audio segment
        extracted_segment = audio_segment[start_frame:end_frame]

        # Convert to numpy array for dB calculation
        # samples = np.array(extracted_segment.get_array_of_samples())
        rms_amplitude = np.sqrt(np.mean(extracted_segment.astype(float) ** 2))

        # Determine the reference amplitude based on the bit depth
        # hardcoding 16-bit depth since we are normalizing the audio to 16-bit
        bit_depth = 16
        reference_amplitude = 2 ** (bit_depth - 1)

        # Calculate dB level
        return 20 * np.log10(rms_amplitude / reference_amplitude)
