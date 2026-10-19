"""Tests for the SPD synthesizer-section to VCE Inventory inverse."""
import os
import subprocess
import struct
import sys
import tempfile
import unittest

from tools.spd2vce import build_compound_file, build_inventory


def make_section():
    sec = bytearray(0x4B0)
    struct.pack_into("<4I", sec, 0, 16000, 1, 1, 0)
    struct.pack_into("<3I", sec, 0x14, 2, 0x4B0, 2)
    sec[0x4B0:0x4B0 + 16] = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    struct.pack_into("<3I", sec, 0x194, 4, 0x4C0, 1)
    sec[0x4C0:0x4D0] = bytes(range(16))
    sec.extend(b"\0" * (0x4F5 - len(sec)))
    struct.pack_into("<5I", sec, 0x490, 3, 0x4D0, 10, 512, 9)
    struct.pack_into("<4I", sec, 0x4D0, 0x4E0, 0x4EA, 0x4F0, 0x4F5)
    sec[0x4E0:0x4EA] = b"unit-0data"
    sec[0x4EA:0x4F0] = b"unit1!"
    sec[0x4F0:0x4F5] = b"last!"
    return bytes(sec)


def compound_stream(blob):
    """Read the sole stream in the minimal OLE files built by the converter."""
    sector_size = 1 << struct.unpack_from("<H", blob, 30)[0]
    mini_sector_size = 1 << struct.unpack_from("<H", blob, 32)[0]

    def sector(index):
        start = (index + 1) * sector_size
        return blob[start:start + sector_size]

    fat_ids = [x for x in struct.unpack_from("<109I", blob, 76) if x != 0xFFFFFFFF]
    difat_sector, difat_count = struct.unpack_from("<II", blob, 68)
    for _ in range(difat_count):
        values = struct.unpack("<%dI" % (sector_size // 4), sector(difat_sector))
        fat_ids.extend(x for x in values[:-1] if x != 0xFFFFFFFF)
        difat_sector = values[-1]
    fat_ids = fat_ids[:struct.unpack_from("<I", blob, 44)[0]]
    fat = [value for index in fat_ids for value in struct.unpack("<%dI" % (sector_size // 4), sector(index))]

    def read_chain(start, allocation):
        out = bytearray()
        current = start
        while current < 0xFFFFFFF8:
            out.extend(allocation(current))
            current = fat[current]
        return bytes(out)

    directory = read_chain(struct.unpack_from("<I", blob, 48)[0], sector)
    stream = directory[128:256]
    size = struct.unpack_from("<Q", stream, 120)[0]
    start = struct.unpack_from("<I", stream, 116)[0]
    if size >= struct.unpack_from("<I", blob, 56)[0]:
        return read_chain(start, sector)[:size]

    mini_fat_start, mini_fat_count = struct.unpack_from("<II", blob, 60)
    mini_fat_data = b"".join(sector(mini_fat_start + i) for i in range(mini_fat_count))
    mini_fat = struct.unpack("<%dI" % (len(mini_fat_data) // 4), mini_fat_data)
    root = directory[:128]
    root_start = struct.unpack_from("<I", root, 116)[0]
    mini_stream = read_chain(root_start, sector)
    out = bytearray()
    current = start
    while current < 0xFFFFFFF8:
        offset = current * mini_sector_size
        out.extend(mini_stream[offset:offset + mini_sector_size])
        current = mini_fat[current]
    return bytes(out[:size])


class Spd2VceTests(unittest.TestCase):
    def test_inventory_roundtrip_layout(self):
        inventory = build_inventory(make_section())
        expected = (struct.pack("<7I", 16000, 10, 1, 1, 0, 512, 9)
                    + struct.pack("<2I", 2, 2) + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
                    + struct.pack("<2I", 1, 4) + bytes(range(16))
                    + struct.pack("<2I", 3, 21) + struct.pack("<4I", 0, 10, 16, 21)
                    + b"unit-0dataunit1!last!")
        self.assertEqual(inventory, expected)

    def test_compound_file_keeps_small_and_regular_streams(self):
        for payload in (b"small Inventory", b"regular Inventory" * 400,
                        bytes(range(256)) * 28000):
            with self.subTest(size=len(payload)):
                result = build_compound_file(payload)
                self.assertEqual(result[:8], bytes.fromhex("D0CF11E0A1B11AE1"))
                self.assertEqual((len(result) - 512) % 512, 0)
                self.assertEqual(compound_stream(result), payload)

    def test_cli_writes_inventory_ole_file(self):
        section = make_section()
        spd = bytearray(0x48) + section
        spd[:8] = b"Data" + struct.pack("<I", 0x10000)
        struct.pack_into("<II", spd, 0x18 + 3 * 8, 0x48, len(section))
        expected = build_inventory(section)
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.spd")
            output = os.path.join(tmp, "output.vce")
            with open(source, "wb") as f:
                f.write(spd)
            script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "spd2vce.py")
            result = subprocess.run([sys.executable, script, source, output],
                                    check=True, capture_output=True)
            with open(output, "rb") as f:
                self.assertEqual(compound_stream(f.read()), expected)
            self.assertIn(b"3 units", result.stdout)


if __name__ == "__main__":
    unittest.main()
