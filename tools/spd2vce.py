"""Convert the synthesizer data in a SAPI 5 .spd file to a SAPI 4 .vce file.

Only the .vce Inventory stream can be reconstructed from an .spd file. The other
SAPI 4 streams and the .spd phone set, trees, durations and amplitudes have no
counterpart in the other format. The output is an OLE compound file containing
the reconstructed Inventory stream; it is suitable for round-tripping through
tools/vce2spd.py, but is not a complete SAPI 4 voice.

    python tools/spd2vce.py Mike.spd Mike.vce
"""
import argparse
import math
import struct


U32_MAX = 0xFFFFFFFF
FREESECT = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE
FATSECT = 0xFFFFFFFD
DIFSECT = 0xFFFFFFFC
SECTOR_SIZE = 512
MINI_SECTOR_SIZE = 64
MINI_STREAM_CUTOFF = 4096


def spd_sections(spd):
    if len(spd) < 0x48 or spd[:4] != b"Data" or struct.unpack_from("<I", spd, 4)[0] != 0x10000:
        raise ValueError("not a v1 .spd file")
    return [struct.unpack_from("<II", spd, 0x18 + 8 * i) for i in range(6)]


def build_inventory(sec3):
    """Repack an SPD synthesizer section into the VCE Inventory stream format."""
    if len(sec3) < 0x4B0:
        raise ValueError("synthesizer section is shorter than its header")

    def u32(offset):
        if offset < 0 or offset + 4 > len(sec3):
            raise ValueError("synthesizer header is truncated")
        return struct.unpack_from("<I", sec3, offset)[0]

    rate = u32(0)
    n_lsf, n_exc, n_dexc = u32(4), u32(8), u32(0xC)
    order, fftn, log2 = u32(0x498), u32(0x49C), u32(0x4A0)
    book_specs = (("lsf", n_lsf, 0x14), ("exc", n_exc, 0x194), ("dexc", n_dexc, 0x314))
    for name, count, _ in book_specs:
        if count > 32:
            raise ValueError("%s codebook count exceeds the synthesizer header capacity" % name)

    inventory = bytearray(struct.pack("<7I", rate, order, n_lsf, n_exc, n_dexc, fftn, log2))
    for name, count, table in book_specs:
        for i in range(count):
            dim, offset, entries = struct.unpack_from("<3I", sec3, table + 12 * i)
            size = dim * entries * 4
            if offset < 0x4B0 or offset + size > len(sec3):
                raise ValueError("%s codebook %d is outside the synthesizer section" % (name, i))
            inventory.extend(struct.pack("<2I", entries, dim))
            inventory.extend(sec3[offset:offset + size])

    n_units, table = u32(0x490), u32(0x494)
    if table < 0x4B0 or table + 4 * (n_units + 1) > len(sec3):
        raise ValueError("unit offset table is outside the synthesizer section")
    offsets = struct.unpack_from("<%dI" % (n_units + 1), sec3, table)
    if offsets[0] < table + 4 * (n_units + 1) or offsets[-1] > len(sec3):
        raise ValueError("unit data is outside the synthesizer section")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("unit offsets are not monotonically increasing")

    unit_data = b"".join(sec3[a:b] for a, b in zip(offsets, offsets[1:]))
    if len(unit_data) > U32_MAX:
        raise ValueError("unit data exceeds the VCE format limit")
    inventory.extend(struct.pack("<2I", n_units, len(unit_data)))
    relative = [0]
    for a, b in zip(offsets, offsets[1:]):
        relative.append(relative[-1] + b - a)
    inventory.extend(struct.pack("<%dI" % len(relative), *relative))
    inventory.extend(unit_data)
    if len(inventory) > U32_MAX:
        raise ValueError("Inventory stream exceeds the VCE format limit")
    return bytes(inventory)


def _directory_entry(name, entry_type, child, start, size):
    entry = bytearray(128)
    encoded = (name + "\0").encode("utf-16le")
    if len(encoded) > 64:
        raise ValueError("OLE directory entry name is too long")
    entry[:len(encoded)] = encoded
    struct.pack_into("<HBB", entry, 64, len(encoded), entry_type, 1)
    struct.pack_into("<III", entry, 68, FREESECT, FREESECT, child)
    struct.pack_into("<I", entry, 116, start)
    struct.pack_into("<Q", entry, 120, size)
    return entry


def build_compound_file(payload, stream_name="Inventory"):
    """Build a minimal v3 OLE compound file containing one stream."""
    if len(payload) > U32_MAX:
        raise ValueError("OLE v3 streams are limited to 4 GiB")

    sectors = []
    chains = []
    directory_sector = 0
    sectors.append(bytearray(SECTOR_SIZE))
    if len(payload) >= MINI_STREAM_CUTOFF:
        stream_sector = len(sectors)
        stream_count = math.ceil(len(payload) / SECTOR_SIZE)
        for i in range(stream_count):
            sectors.append(bytearray(payload[i * SECTOR_SIZE:(i + 1) * SECTOR_SIZE].ljust(SECTOR_SIZE, b"\0")))
        chains.append((stream_sector, stream_count))
        mini_fat_sector = FREESECT
        mini_fat_count = 0
        mini_stream_sector = ENDOFCHAIN
        mini_stream_size = 0
        stream_start = stream_sector
    else:
        mini_count = math.ceil(len(payload) / MINI_SECTOR_SIZE)
        mini_data = payload.ljust(mini_count * MINI_SECTOR_SIZE, b"\0")
        mini_fat = [FREESECT] * (SECTOR_SIZE // 4)
        for i in range(mini_count):
            mini_fat[i] = i + 1 if i + 1 < mini_count else ENDOFCHAIN
        mini_fat_bytes = struct.pack("<%dI" % len(mini_fat), *mini_fat)
        mini_fat_sector = len(sectors)
        sectors.append(bytearray(mini_fat_bytes))
        mini_fat_count = 1
        chains.append((mini_fat_sector, mini_fat_count))
        mini_stream_sector = len(sectors)
        mini_stream_size = len(mini_data)
        for i in range(0, len(mini_data), SECTOR_SIZE):
            sectors.append(bytearray(mini_data[i:i + SECTOR_SIZE].ljust(SECTOR_SIZE, b"\0")))
        mini_stream_count = len(sectors) - mini_stream_sector
        if mini_stream_count:
            chains.append((mini_stream_sector, mini_stream_count))
        stream_start = 0 if mini_count else ENDOFCHAIN

    directory = bytearray(SECTOR_SIZE)
    directory[:128] = _directory_entry("Root Entry", 5, 1, mini_stream_sector, mini_stream_size)
    directory[128:256] = _directory_entry(stream_name, 2, FREESECT, stream_start, len(payload))
    sectors[directory_sector] = directory
    chains.append((directory_sector, 1))

    base_sector_count = len(sectors)
    fat_count, difat_count = 1, 0
    while True:
        new_difat_count = max(0, math.ceil((fat_count - 109) / 127))
        new_fat_count = math.ceil((base_sector_count + new_difat_count + fat_count) / (SECTOR_SIZE // 4))
        if (new_fat_count, new_difat_count) == (fat_count, difat_count):
            break
        fat_count, difat_count = new_fat_count, new_difat_count
    difat_start = base_sector_count
    fat_start = difat_start + difat_count
    difat_sectors = list(range(difat_start, fat_start))
    fat_sectors = list(range(fat_start, fat_start + fat_count))
    sectors.extend(bytearray(SECTOR_SIZE) for _ in range(difat_count + fat_count))

    fat = [FREESECT] * (len(sectors) * (SECTOR_SIZE // 4))

    def mark_chain(start, count):
        for i in range(count):
            fat[start + i] = start + i + 1 if i + 1 < count else ENDOFCHAIN

    for start, count in chains:
        mark_chain(start, count)
    for sector in difat_sectors:
        fat[sector] = DIFSECT
    for sector in fat_sectors:
        fat[sector] = FATSECT

    fat_bytes = struct.pack("<%dI" % (len(fat_sectors) * SECTOR_SIZE // 4),
                            *(fat[:len(fat_sectors) * SECTOR_SIZE // 4]))
    for i, sector in enumerate(fat_sectors):
        sectors[sector][:] = fat_bytes[i * SECTOR_SIZE:(i + 1) * SECTOR_SIZE]

    if difat_count:
        remaining_fat = fat_sectors[109:]
        for i, sector in enumerate(difat_sectors):
            block = remaining_fat[i * 127:(i + 1) * 127]
            values = block + [FREESECT] * (127 - len(block))
            values.append(difat_sectors[i + 1] if i + 1 < difat_count else ENDOFCHAIN)
            sectors[sector][:] = struct.pack("<128I", *values)

    header = bytearray(SECTOR_SIZE)
    header[:8] = bytes.fromhex("D0CF11E0A1B11AE1")
    struct.pack_into("<HHHHH", header, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<I", header, 40, 0)
    struct.pack_into("<I", header, 44, fat_count)
    struct.pack_into("<I", header, 48, directory_sector)
    struct.pack_into("<I", header, 56, MINI_STREAM_CUTOFF)
    struct.pack_into("<II", header, 60, mini_fat_sector, mini_fat_count)
    struct.pack_into("<II", header, 68, difat_sectors[0] if difat_sectors else ENDOFCHAIN, difat_count)
    difat = fat_sectors[:109] + [FREESECT] * (109 - min(len(fat_sectors), 109))
    struct.pack_into("<109I", header, 76, *difat)
    return bytes(header) + b"".join(sectors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spd", help="source SAPI 5 .spd file")
    ap.add_argument("out", help="output Inventory-only SAPI 4 .vce file")
    args = ap.parse_args()
    spd = open(args.spd, "rb").read()
    try:
        sections = spd_sections(spd)
    except ValueError as exc:
        ap.error(str(exc))
    offset, size = sections[3]
    if offset < 0x48 or offset + size > len(spd):
        ap.error("synthesizer section is outside the .spd file")
    try:
        inventory = build_inventory(spd[offset:offset + size])
        vce = build_compound_file(inventory)
    except (struct.error, ValueError) as exc:
        ap.error(str(exc))
    with open(args.out, "wb") as f:
        f.write(vce)
    print("wrote %s (%d-byte Inventory; %d units)" %
          (args.out, len(inventory), struct.unpack_from("<I", spd[offset + 0x490:offset + 0x494])[0]))


if __name__ == "__main__":
    main()
