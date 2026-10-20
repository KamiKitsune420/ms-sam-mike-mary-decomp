"""Offline BigVGAN synthesis using the checkpoint's waveform-to-mel frontend."""

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave


HEADER = struct.Struct("<4sIII")
RECORD = struct.Struct("<IIf")


def require_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("NumPy is required; install the BigVGAN environment requirements") from exc
    return numpy


def read_lpc_silence_ranges(path):
    silence_ranges = []
    total_samples = 0
    with open(path, "rb") as f:
        header = f.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("LPC feature file has no complete header")
        magic, version, sample_rate, order = HEADER.unpack(header)
        if magic != b"SLPC" or version != 1 or sample_rate <= 0 or order <= 0:
            raise ValueError("unsupported LPC feature file")
        while True:
            raw = f.read(RECORD.size)
            if not raw:
                break
            if len(raw) != RECORD.size:
                raise ValueError("truncated LPC frame record")
            samples, flags, _amp = RECORD.unpack(raw)
            coeffs_raw = f.read(4 * (order + 1))
            if len(coeffs_raw) != 4 * (order + 1):
                raise ValueError("truncated LPC coefficients")
            if flags not in (0, 1, 2):
                raise ValueError("invalid LPC frame flags")
            if samples and flags == 0:
                silence_ranges.append((total_samples, total_samples + samples))
            total_samples += samples
    if total_samples == 0:
        raise ValueError("LPC feature file contains no audio frames")
    return sample_rate, total_samples, silence_ranges


def load_bigvgan(checkpoint, repo, device):
    sys.path.insert(0, str(repo))
    try:
        import torch
        import bigvgan
    except ImportError as exc:
        raise RuntimeError(
            "BigVGAN dependencies are unavailable; install the requirements from the BigVGAN repository"
        ) from exc

    checkpoint = pathlib.Path(checkpoint)
    if checkpoint.is_dir():
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "bigvgan_generator.pt"
    else:
        weights_path = checkpoint
        config_path = checkpoint.parent / "config.json"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("checkpoint needs config.json and bigvgan_generator.pt")
    h = bigvgan.load_hparams_from_json(str(config_path))
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    checkpoint_data = torch.load(str(weights_path), map_location="cpu")
    weights = checkpoint_data.get("generator", checkpoint_data)
    model.load_state_dict(weights)
    model.remove_weight_norm()
    model.eval().to(device)
    return torch, model, h


def reference_to_mels(path, h, torch, device):
    np = require_numpy()
    try:
        import librosa
        from meldataset import get_mel_spectrogram
    except ImportError as exc:
        raise RuntimeError(
            "BigVGAN mel dependencies are unavailable; install the BigVGAN environment requirements"
        ) from exc

    target_rate = int(h.sampling_rate)
    hop = int(h.hop_size)
    if target_rate <= 0 or hop <= 0:
        raise ValueError("invalid BigVGAN STFT parameters in checkpoint config")
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("sam_say reference WAV must be 16-bit mono PCM")
        source_rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if source_rate <= 0 or audio.size == 0:
        raise ValueError("sam_say produced an empty or invalid reference WAV")
    source_samples = audio.size
    if source_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate)

    audio = librosa.util.normalize(audio) * 0.95
    target_samples = audio.size - audio.size % hop
    if target_samples <= 0:
        raise ValueError("speech is shorter than one BigVGAN hop")
    audio = np.ascontiguousarray(audio[:target_samples], dtype=np.float32)
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    mel = get_mel_spectrogram(waveform, h)
    return mel, target_rate, target_samples, source_rate, source_samples


def write_wav(path, audio, sample_rate, target_samples, source_rate, silence_ranges):
    np = require_numpy()
    if audio.size != target_samples:
        raise ValueError(f"BigVGAN returned {audio.size} samples; expected {target_samples}")
    for start, end in silence_ranges:
        lo = int(start * sample_rate / source_rate)
        hi = min(target_samples, int(end * sample_rate / source_rate))
        audio[lo:hi] = 0.0
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-say", required=True, help="path to the built sam_say executable")
    parser.add_argument("--data", required=True, help="voice data directory")
    parser.add_argument("--voice", default="Sam")
    parser.add_argument("--bigvgan-repo", required=True, help="path to the NVIDIA BigVGAN checkout")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or generator .pt file")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("text", help="text to speak, or @path to a UTF-8 text file")
    parser.add_argument("output", help="output WAV path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="sam-bigvgan-") as temp:
        features = pathlib.Path(temp) / "speech.slpc"
        reference = pathlib.Path(temp) / "sam.wav"
        command = [
            args.sam_say,
            "--data",
            args.data,
            "--voice",
            args.voice,
            "--vocoder",
            "sam",
            "--dump-lpc",
            str(features),
            args.text,
            str(reference),
        ]
        subprocess.run(command, check=True)
        source_rate, source_samples, silence_ranges = read_lpc_silence_ranges(features)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; install the BigVGAN environment requirements") from exc
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
        torch, model, h = load_bigvgan(args.checkpoint, args.bigvgan_repo, device)
        mel, output_rate, target_samples, reference_rate, reference_samples = reference_to_mels(
            reference, h, torch, device
        )
        if (source_rate, source_samples) != (reference_rate, reference_samples):
            raise RuntimeError("SAM audio and exported LPC timing do not match")
        with torch.inference_mode():
            generated = model(mel).reshape(-1).float().cpu().numpy()
        if not require_numpy().isfinite(generated).all():
            raise RuntimeError("BigVGAN generated non-finite audio samples")
        write_wav(args.output, generated, output_rate, target_samples, source_rate, silence_ranges)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"bigvgan_say: {exc}", file=sys.stderr)
        raise SystemExit(1)
