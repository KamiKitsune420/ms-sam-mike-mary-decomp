"""Convert a SAPI 4 voice (.vce) into the SAPI 5 voice file (.spd) this engine reads.

The two are the same voice data in different wrappers. A .vce is an OLE compound file whose Inventory
stream holds the synthesizer data; a .spd is a five-section container whose section 3 holds the same
thing. The codebooks and the unit data are byte for byte identical - only the packing differs:

  .vce Inventory: 7 words (rate, order, n_lsf, n_exc, n_dexc, fft, log2), then every codebook inline as
                  count, dim, floats; then n_units, the size of the unit data, a table of offsets
                  RELATIVE to the data that follows it, then the unit blobs.
  .spd section 3: a fixed 0x4B0 header of (dim, offset, count) triples pointing at the codebooks, the
                  sine / window / noise tables and a unit table of ABSOLUTE offsets.

Sections 0, 1 and 2 (phone set, per-leaf durations and amplitudes, the unit trees) have no direct
equivalent in the .vce - they are built from its Senone / TreeImage / PhoneFile streams - so they are
taken from a donor .spd with the same unit count (Sam 3365; Mike, Mary and the rest 3333).

`tools/spd2vce.py` reverses the Inventory/section-3 packing. Its output contains only the reconstructed
Inventory stream: the other SAPI 4 streams cannot be recovered from an .spd.

  python tools/vce2spd.py freddy.vce Freddy.spd --donor Mike.spd
  python tools/vce2spd.py sam.vce out.spd --donor Sam.spd --verify Sam.spd
  python tools/spd2vce.py Mike.spd Mike.vce

Both files are Microsoft's voice data: the result is for your own machines.
"""
import argparse
import os
import struct
import sys

try:
    import olefile
except ImportError:
    sys.exit("needs olefile:  pip install olefile")


def read_inventory(path):
    """-> dict with the header fields, the codebooks and the unit blobs"""
    ole = olefile.OleFileIO(path)
    if not ole.exists("Inventory"):
        sys.exit("%s has no Inventory stream - not a voice?" % path)
    inv = ole.openstream("Inventory").read()
    ole.close()
    rate, order, nlsf, nexc, ndexc, fftn, log2 = struct.unpack_from("<7I", inv, 0)
    p = 28
    books = {"lsf": [], "exc": [], "dexc": []}
    for kind, n in (("lsf", nlsf), ("exc", nexc), ("dexc", ndexc)):
        for _ in range(n):
            cnt, dim = struct.unpack_from("<2I", inv, p)
            p += 8
            books[kind].append((dim, cnt, inv[p:p + 4 * cnt * dim]))
            p += 4 * cnt * dim
    nunits, datasz = struct.unpack_from("<2I", inv, p)
    tab = p + 8
    base = tab + 4 * (nunits + 1)
    offs = struct.unpack_from("<%dI" % (nunits + 1), inv, tab)
    units = [inv[base + offs[i]:base + offs[i + 1]] for i in range(nunits)]
    return dict(rate=rate, order=order, fftn=fftn, log2=log2, books=books, units=units, datasz=datasz)


def spd_sections(spd):
    return [struct.unpack_from("<II", spd, 0x18 + 8 * i) for i in range(6)]


def build_section3(v, donor_sec3):
    """the .vce inventory in .spd section 3 layout; the DSP tables come from the donor"""
    hdr = bytearray(0x4B0)
    body = bytearray()

    def put(data):
        off = 0x4B0 + len(body)
        body.extend(data)
        return off

    def setu32(o, val):
        struct.pack_into("<I", hdr, o, val & 0xFFFFFFFF)

    setu32(0x00, v["rate"])
    setu32(0x04, len(v["books"]["lsf"]))
    setu32(0x08, len(v["books"]["exc"]))
    setu32(0x0C, len(v["books"]["dexc"]))
    for tabo, kind in ((0x14, "lsf"), (0x194, "exc"), (0x314, "dexc")):
        for i, (dim, cnt, data) in enumerate(v["books"][kind]):
            off = put(data)
            setu32(tabo + 12 * i, dim)
            setu32(tabo + 12 * i + 4, off)
            setu32(tabo + 12 * i + 8, cnt)
    # the sine, window and noise tables are not in the .vce; copy the donor's (same engine, same sizes)
    d = donor_sec3
    for src, dstoff, count in ((struct.unpack_from("<I", d, 0x4A4)[0], 0x4A4, 2 * v["fftn"]),
                               (struct.unpack_from("<I", d, 0x4A8)[0], 0x4A8, v["fftn"]),
                               (struct.unpack_from("<I", d, 0x4AC)[0], 0x4AC, v["rate"])):
        setu32(dstoff, put(d[src:src + 4 * count]))
    setu32(0x490, len(v["units"]))
    setu32(0x498, v["order"])
    setu32(0x49C, v["fftn"])
    setu32(0x4A0, v["log2"])
    setu32(0x10, struct.unpack_from("<I", d, 0x10)[0])          # unused by the loader, kept for fidelity
    tab_off = 0x4B0 + len(body)
    setu32(0x494, tab_off)
    body.extend(b"\0" * 4 * (len(v["units"]) + 1))
    offs = []
    for blob in v["units"]:
        offs.append(0x4B0 + len(body))
        body.extend(blob)
    for i, o in enumerate(offs):
        struct.pack_into("<I", body, tab_off - 0x4B0 + 4 * i, o)
    struct.pack_into("<I", body, tab_off - 0x4B0 + 4 * len(offs), 0x4B0 + len(body))
    return bytes(hdr) + bytes(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vce")
    ap.add_argument("out")
    ap.add_argument("--donor", required=True, help="a .spd with the same unit count, for sections 0/1/2")
    ap.add_argument("--verify", help="a .spd this conversion should reproduce (for testing the converter)")
    a = ap.parse_args()

    v = read_inventory(a.vce)
    donor = open(a.donor, "rb").read()
    sec = spd_sections(donor)
    doff3, dsz3 = sec[3]
    dsec3 = donor[doff3:doff3 + dsz3]
    dunits = struct.unpack_from("<I", dsec3, 0x490)[0]
    print("%s: %d Hz, order %d, %d units" % (os.path.basename(a.vce), v["rate"], v["order"], len(v["units"])))
    print("%s: %d units" % (os.path.basename(a.donor), dunits))
    if dunits != len(v["units"]):
        sys.exit("donor has %d units, the voice has %d - pick a donor with the same count "
                 "(Sam.spd for 3365, Mike.spd or Mary.spd for 3333)" % (dunits, len(v["units"])))

    new3 = build_section3(v, dsec3)
    out = bytearray(donor[:doff3]) + new3 + donor[doff3 + dsz3:]
    struct.pack_into("<II", out, 0x18 + 24, doff3, len(new3))
    shift = len(new3) - dsz3
    for i in range(6):                                   # sections stored after ours move
        o, s = struct.unpack_from("<II", out, 0x18 + 8 * i)
        if i != 3 and o > doff3:
            struct.pack_into("<I", out, 0x18 + 8 * i, o + shift)
    open(a.out, "wb").write(bytes(out))
    print("wrote %s (%.2f MB, section 3 %d -> %d bytes)" % (a.out, len(out) / 1e6, dsz3, len(new3)))

    sdf = os.path.splitext(a.donor)[0] + ".sdf"
    if os.path.exists(sdf):
        import shutil
        shutil.copy2(sdf, os.path.splitext(a.out)[0] + ".sdf")
        print("copied %s for the base pitch" % os.path.basename(sdf))

    if a.verify:
        ref = open(a.verify, "rb").read()
        roff3, rsz3 = spd_sections(ref)[3]
        r3 = ref[roff3:roff3 + rsz3]
        n3 = new3
        print("verify against %s:" % os.path.basename(a.verify))
        same = sum(1 for i in range(min(len(r3), len(n3))) if r3[i] == n3[i])
        print("  section 3: %d vs %d bytes, %d identical bytes" % (len(r3), len(n3), same))
        rt = struct.unpack_from("<I", r3, 0x494)[0]
        nt = struct.unpack_from("<I", n3, 0x494)[0]
        ro = struct.unpack_from("<%dI" % (dunits + 1), r3, rt)
        no = struct.unpack_from("<%dI" % (dunits + 1), n3, nt)
        ok = sum(1 for i in range(dunits) if r3[ro[i]:ro[i + 1]] == n3[no[i]:no[i + 1]])
        print("  units identical: %d of %d" % (ok, dunits))


if __name__ == "__main__":
    main()
