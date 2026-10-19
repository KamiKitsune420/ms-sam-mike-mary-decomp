"""Tests for WAV and HTK-lab input to the custom SPD voice builder."""
import os
import struct
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from annavoice.spdbuild import estimate_period, load_wav, make_pitch_epochs, parse_htk_lab, stage_wav


class SpdBuildWavTests(unittest.TestCase):
    def test_parse_htk_phone_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "phones.lab")
            with open(lab, "w", encoding="ascii") as f:
                f.write("0 5000000 sil\n")
                f.write("5000000 15000000 AA\n")
                f.write("15000000 20000000 T\n")
            labels = parse_htk_lab(lab, 32000)
        self.assertEqual(labels, [
            (0, 8000, "SIL", ["SIL"]),
            (8000, 24000, "AA", ["AA0", "AA1"]),
            (24000, 32000, "T", ["T"]),
        ])

    def test_rejects_overlapping_and_out_of_range_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "invalid.lab")
            with open(lab, "w", encoding="ascii") as f:
                f.write("0 5000000 AA1\n")
                f.write("4000000 6000000 T\n")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                parse_htk_lab(lab, 16000)

            with open(lab, "w", encoding="ascii") as f:
                f.write("10000000 11000000 T\n")
            with self.assertRaisesRegex(ValueError, "after the end"):
                parse_htk_lab(lab, 16000)

    def test_period_estimation_and_signed_epoch_generation(self):
        rate = 16000
        t = np.arange(rate // 2) / rate
        audio = 0.5 * np.sin(2 * np.pi * 100 * t)
        self.assertAlmostEqual(estimate_period(audio, 4000), 160, delta=2)
        epochs = make_pitch_epochs(audio)
        self.assertGreater(len(epochs), 10)
        self.assertTrue(np.all(epochs > 0))
        self.assertLessEqual(abs(int(np.abs(epochs).sum()) - len(audio)), 320)

    def test_silence_is_unvoiced(self):
        self.assertIsNone(estimate_period(np.zeros(800), 400))

    def test_wav_loader_mixes_and_resamples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stereo.wav")
            source = np.full((800, 2), 8192, dtype=np.int16)
            wavfile.write(path, 8000, source)
            audio = load_wav(path)
        self.assertEqual(len(audio), 1600)
        self.assertTrue(np.allclose(audio[100:-100], 0.25, atol=1e-3))

    def test_wav_loader_centers_unsigned_pcm8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pcm8.wav")
            wavfile.write(path, 16000, np.full(800, 128, dtype=np.uint8))
            audio = load_wav(path)
        self.assertTrue(np.allclose(audio, 0.0))

    def test_wav_command_builds_spd_from_minimal_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            leaf_path = os.path.join(tmp, "samleaf.bin")
            names = ["SIL", "AA1"]
            table = np.empty((2, 2, 2, 4), dtype="<i4")
            table[0] = 0
            table[1] = 1
            with open(leaf_path, "wb") as f:
                f.write(b"SLF1")
                f.write(struct.pack("<2i", 2, 2))
                f.write(table.tobytes())
                f.write(np.array([0, 1], dtype="<i4").tobytes())
                f.write("\0".join(names).encode("ascii") + b"\0")

            synth_offset = 0x50
            sam = bytearray(synth_offset + 0x4B0)
            sam[:8] = b"Data" + struct.pack("<I", 0x10000)
            struct.pack_into("<II", sam, 0x18 + 8, 0x48, 8)
            struct.pack_into("<II", sam, 0x18 + 24, synth_offset, 0x4B0)
            struct.pack_into("<I", sam, synth_offset + 0x498, 20)
            struct.pack_into("<II", sam, synth_offset + 0x49C, 512, 9)
            sam_path = os.path.join(tmp, "Sam.spd")
            with open(sam_path, "wb") as f:
                f.write(sam)

            rate = 16000
            audio = np.zeros(rate, dtype=np.int16)
            t = np.arange(12000) / rate
            audio[4000:] = (np.sin(2 * np.pi * 120 * t[:12000]) * 12000).astype(np.int16)
            wav_path = os.path.join(tmp, "recording.wav")
            wavfile.write(wav_path, rate, audio)
            lab_path = os.path.join(tmp, "recording.lab")
            with open(lab_path, "w", encoding="ascii") as f:
                f.write("0 2500000 sil\n")
                f.write("2500000 10000000 AA1\n")
            output_path = os.path.join(tmp, "Custom.spd")

            stage_wav(type("Args", (), {
                "wav": wav_path, "lab": lab_path, "leaf_table": leaf_path, "sam": sam_path,
                "out": output_path, "rate": rate, "books": "fine", "per_unit": 1,
                "max_frames": 48, "keep_durations": True,
            })())

            with open(output_path, "rb") as f:
                result = f.read()
            out_offset, out_size = struct.unpack_from("<II", result, 0x18 + 24)
            self.assertEqual(struct.unpack_from("<I", result, out_offset + 0x490)[0], 2)
            self.assertGreater(out_size, 0x4B0)


if __name__ == "__main__":
    unittest.main()
