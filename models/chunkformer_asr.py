import pathlib
import numpy as np
from typing import List, Union, Optional, NamedTuple
import torch
from tqdm import tqdm
import torchaudio.compliance.kaldi as kaldi
from pydub import AudioSegment
from typing import TypedDict
from chunkformer.decode import init

def load_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # set bit depth to 16bit
    audio = audio.set_channels(1)  # set to mono
    audio = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
    return audio

class SingleWordSegment(TypedDict):
    """
    A single word of a speech.
    """
    word: str
    start: float
    end: float
    score: float

class SingleAlignedSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """
    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    #chars: Optional[List[SingleCharSegment]]


import math
from typing import List, Tuple

def remove_duplicates_and_blank(hyp: List[Tuple[int, int, int]]) -> Tuple[List[int], List[int], List[int]]:
    idxs: List[int] = []
    new_hyp: List[int] = []
    probs: List[int] = []
    cur = 0
    while cur < len(hyp):
        time_stamp, token, prob = hyp[cur]
        if token != 0:
            idxs.append(time_stamp)
            new_hyp.append(token)
            probs.append(prob)
        prev_token = token
        cur += 1
        while cur < len(hyp) and hyp[cur][0] == prev_token:
            cur += 1
    return idxs, new_hyp, probs

def class2str(target, char_dict):
    content = []
    for w in target:
        content.append(char_dict[int(w)])
    return ''.join(content).replace('▁',' ').strip()

def token_idx_to_seconds(idx: int) -> float:
    return idx * 8 * 10 / 1000  # Convert 80ms per frame to seconds

def get_aligned_segment_output(hyps, probs, char_dict, max_silence = 20):
    decodes = []
    for tokens, _probs in zip(hyps, probs):
        tokens = tokens.cpu()
        start = -1
        end = -1
        prev_end = -1
        silence_cum = 0
        decode_per_time = []  # Now stores tuples of (token, time_stamp)
        decode = []
        for time_stamp, (token, prob) in enumerate(zip(tokens, _probs)):
            if token == 0:
                silence_cum += 1
            else:
                if start == -1 and end == -1:
                    if prev_end != -1:
                        start = math.ceil((time_stamp + prev_end) / 2)
                    else:
                        start = max(time_stamp - int(max_silence/2), 0)
                silence_cum = 0
                decode_per_time.append((time_stamp, token, prob))

            if silence_cum == max_silence and start != -1:
                end = time_stamp
                prev_end = end
                idxs, hyps_tokens, hyps_probs = remove_duplicates_and_blank(decode_per_time)
                item = SingleAlignedSegment(
                    start=token_idx_to_seconds(start),
                    end=token_idx_to_seconds(end),
                    text=class2str(hyps_tokens, char_dict),
                    words=[
                        SingleWordSegment(
                            word=class2str([token], char_dict).lstrip(),
                            start=token_idx_to_seconds(ts),
                            end=token_idx_to_seconds(ts + 1),
                            score=prob.item()
                        )
                        for ts, token, prob in zip(idxs, hyps_tokens, hyps_probs)
                    ]
                )
                decode.append(item)
                decode_per_time = []
                start = -1
                end = -1
                silence_cum = 0

        if start != -1 and end == -1 and len(decode_per_time) > 0:
            idxs, hyps_tokens, hyps_probs = remove_duplicates_and_blank(decode_per_time)
            item = SingleAlignedSegment(
                start=token_idx_to_seconds(start),
                end=token_idx_to_seconds(time_stamp),
                text=class2str(hyps_tokens, char_dict),
                words=[
                    SingleWordSegment(
                        word=class2str([token], char_dict).lstrip(),
                        start=token_idx_to_seconds(ts),
                        end=token_idx_to_seconds(ts + 1),
                        score=prob.item()
                    )
                    for ts, token, prob in zip(idxs, hyps_tokens, hyps_probs)
                ]
            )
            decode.append(item)
        decodes.append(decode)
    return decodes

class ChunkFormerModel:
    def __init__(
        self,
        model_checkpoint,
        device: Union[int, str, "torch.device"] = -1,
        total_batch_duration=1800,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        full_attn=False,
        max_silence=10 # max frames of silece (80ms each) before splitting. Default is 10 (800ms)
    ):
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.char_dict = {}
        self.device = torch.device(device)

        self.total_batch_duration = total_batch_duration
        self.chunk_size = chunk_size
        self.left_context_size = left_context_size
        self.right_context_size = right_context_size
        self.full_attn = full_attn
        self.max_silence = max_silence

    @torch.no_grad
    def ctc_forward(self, xs, xs_lens=None, n_chunks=None):
        ctc_probs = self.model.encoder.ctc.log_softmax(xs)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        hyps = topk_index.squeeze(-1)  # (B, maxlen)
        probs = topk_prob.squeeze(-1).exp() # (B, maxlen), normalized to softmax scores

        if (n_chunks is not None) and (xs_lens is not None):
            hyps = hyps.split(n_chunks, dim=0)
            hyps = [hyp.flatten()[:x_len] for hyp, x_len in zip(hyps, xs_lens)]

            probs = probs.split(n_chunks, dim=0)
            probs = [prob.flatten()[:x_len] for prob, x_len in zip(probs, xs_lens)]
        return hyps, probs

    @torch.no_grad
    def transcribe_and_align(self, audio: np.ndarray):
        def get_max_input_context(c, r, n):
            return r + max(c, r) * (n-1)

        # model configuration
        subsampling_factor = self.model.encoder.embed.subsampling_factor
        chunk_size = self.chunk_size
        left_context_size = self.left_context_size
        right_context_size = self.right_context_size
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        # get the maximum length that the gpu can consume
        max_length_limited_context = self.total_batch_duration
        max_length_limited_context = int((max_length_limited_context // 0.01))//2 # in 10ms second

        multiply_n = max_length_limited_context // chunk_size // subsampling_factor
        truncated_context_size = chunk_size * multiply_n # we only keep this part for text decoding

        # get the relative right context size
        rel_right_context_size = get_max_input_context(chunk_size, max(right_context_size, conv_lorder), self.model.encoder.num_blocks)
        rel_right_context_size = rel_right_context_size * subsampling_factor


        waveform = torch.from_numpy(audio)
        offset = torch.zeros(1, dtype=torch.int, device=self.device)

        # waveform = padding(waveform, sample_rate)
        xs = kaldi.fbank(waveform,
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                dither=0.0,
                                energy_floor=0.0,
                                sample_frequency=16000).unsqueeze(0)
        hyps = []
        probs = []
        att_cache = torch.zeros((self.model.encoder.num_blocks, left_context_size, self.model.encoder.attention_heads, self.model.encoder._output_size * 2 // self.model.encoder.attention_heads)).to(self.device)
        cnn_cache = torch.zeros((self.model.encoder.num_blocks, self.model.encoder._output_size, conv_lorder)).to(self.device)    # print(context_size)
        for idx, _ in tqdm(list(enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)))):
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(truncated_context_size * subsampling_factor * (idx+1) + 7, xs.shape[1])

            x = xs[:, start:end+rel_right_context_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(self.device)

            encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = self.model.encoder.forward_parallel_chunk(xs=x,
                                                                        xs_origin_lens=x_len,
                                                                        chunk_size=chunk_size,
                                                                        left_context_size=left_context_size,
                                                                        right_context_size=right_context_size,
                                                                        att_cache=att_cache,
                                                                        cnn_cache=cnn_cache,
                                                                        truncated_context_size=truncated_context_size,
                                                                        offset=offset
                                                                        )
            encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
            if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
                encoder_outs = encoder_outs[:, :truncated_context_size]  # (B, maxlen, vocab_size) # exclude the output of rel right context
            offset = offset - encoder_lens + encoder_outs.shape[1]


            hyp, prob = self.ctc_forward(encoder_outs)
            hyps.append(hyp.squeeze(0))
            probs.append(prob.squeeze(0))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
                break

        hyps = torch.cat(hyps)
        probs = torch.cat(probs)

        return {"segments": get_aligned_segment_output([hyps], [probs], self.char_dict, self.max_silence)[0]}

    @torch.no_grad
    def transcribe_directory(self, audio_dir: str):
        # enumerate files in audio_dir
        # for each file, load audio, transcribe without alignment
        # return list of transcriptions

        max_length_limited_context = self.total_batch_duration
        max_length_limited_context = int((max_length_limited_context // 0.01)) // 2 # in 10ms second    xs = []
        max_frames = max_length_limited_context
        chunk_size = self.chunk_size
        left_context_size = self.left_context_size
        right_context_size = self.right_context_size

        decodes, prob_decodes = [], []
        xs = []
        xs_origin_lens = []
        hyps, probs = [], []
        audio_paths = list(pathlib.Path(audio_dir).rglob("*.wav"))
        for idx, audio_path in tqdm(list(enumerate(audio_paths)), desc="Transcribing segments"):
            waveform = load_audio(audio_path)
            x = kaldi.fbank(waveform,
                                    num_mel_bins=80,
                                    frame_length=25,
                                    frame_shift=10,
                                    dither=0.0,
                                    energy_floor=0.0,
                                    sample_frequency=16000)

            xs.append(x)
            xs_origin_lens.append(x.shape[0])
            max_frames -= xs_origin_lens[-1]

            if (max_frames <= 0) or (idx == len(audio_paths) - 1):
                xs_origin_lens = torch.tensor(xs_origin_lens, dtype=torch.int, device=self.device)
                offset = torch.zeros(len(xs), dtype=torch.int, device=self.device)
                encoder_outs, encoder_lens, n_chunks, _, _, _ = self.model.encoder.forward_parallel_chunk(xs=xs,
                                                                            xs_origin_lens=xs_origin_lens,
                                                                            chunk_size=chunk_size,
                                                                            left_context_size=left_context_size,
                                                                            right_context_size=right_context_size,
                                                                            offset=offset
                )

                hyps, probs = self.ctc_forward(encoder_outs, encoder_lens, n_chunks)
                decodes += hyps
                prob_decodes += probs

                # reset
                xs = []
                xs_origin_lens = []
                max_frames = max_length_limited_context

        return {
            str(audio_path): {"segments": get_aligned_segment_output([hyps], [probs], self.char_dict, self.max_silence)[0]}
            for audio_path, hyps, probs in zip(audio_paths, decodes, prob_decodes)
        }

    @torch.no_grad
    def load_asr_model(self):
        self.model, self.char_dict = init(self.model_checkpoint, self.device)

    def unload_asr_model(self):
        self.model, self.char_dict = None, None
