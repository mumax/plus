"""Check Binary 4 exports independently of the OVF writer backend."""

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from mumaxplus import FieldQuantity, Grid


@pytest.mark.parametrize("ncomp", [1, 3, 6])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_binary4_payload_and_metadata(tmp_path, ncomp, dtype):
    # A strided input catches component ordering and contiguity mistakes.
    data = np.arange(ncomp * 2 * 3 * 8, dtype=dtype).reshape(ncomp, 2, 3, 8)
    data = (data / 7)[:, :, :, ::2]
    impl = SimpleNamespace(
        name="test:field", unit="A/m", ncomp=ncomp,
        grid=Grid((4, 3, 2), origin=(2, -3, 5))._impl,
        system=SimpleNamespace(cellsize=(4e-9, 2e-9, 5e-9), time=1.25e-9),
        eval=lambda: data,
    )
    quantity = FieldQuantity(impl)
    target = tmp_path / "field.ovf"
    quantity.save_ovf(target)

    with target.open("rb") as stream:
        assert stream.readline() == b"# OOMMF OVF 2.0\n"
        metadata = {}
        while True:
            line = stream.readline()
            assert line, "OVF binary data marker is missing"
            if line == b"# Begin: Data Binary 4\n":
                break
            if b":" in line:
                key, value = line.decode().removeprefix("# ").strip().split(":", 1)
                metadata[key] = value.strip()
        assert struct.unpack("<f", stream.read(4)) == (1234567.0,)
        payload = np.fromfile(stream, dtype="<f4", count=data.size)
        assert stream.read() == b"\n# End: Data Binary 4\n# End: Segment\n"

    expected = np.moveaxis(data, 0, -1).astype(np.float32)
    np.testing.assert_array_equal(payload.reshape(2, 3, 4, ncomp), expected)
    assert metadata["Title"] == "test:field"
    assert metadata["valuedim"] == str(ncomp)
    assert len(metadata["valuelabels"].split()) == ncomp
    assert metadata["valueunits"].split() == ["A/m"] * ncomp
    assert metadata["Desc"] == "Total simulation time: 1.25e-09 s"
    assert [int(metadata[axis + "nodes"]) for axis in "xyz"] == [4, 3, 2]
    np.testing.assert_allclose(
        [float(metadata[axis + "base"]) for axis in "xyz"],
        [8e-9, -6e-9, 25e-9], rtol=1e-15, atol=0,
    )
    np.testing.assert_allclose(
        [float(metadata[axis + "min"]) for axis in "xyz"],
        [6e-9, -7e-9, 22.5e-9], rtol=1e-15, atol=0,
    )
    np.testing.assert_allclose(
        [float(metadata[axis + "max"]) for axis in "xyz"],
        [22e-9, -1e-9, 32.5e-9], rtol=1e-15, atol=0,
    )
