"""Tests for encoder/decoder protocols and ABC helpers."""

import numpy as np
import pytest

from fly_harness.backend import DirectBioSimBackend, InProcessLifBackend
from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import build_reflex_connectome
from fly_harness.protocols import Decoder, Encoder, ModelBackend


def test_reflex_codec_satisfy_protocols() -> None:
    encoder = ReflexEncoder()
    decoder = ReflexDecoder()
    assert isinstance(encoder, Encoder)
    assert isinstance(decoder, Decoder)


def test_reflex_encoder_maps_touch_to_sensory_indices() -> None:
    encoder = ReflexEncoder()
    currents = encoder.encode(TouchObservation(touch_left=1.0, touch_right=0.5))
    assert currents.shape == (24,)
    assert currents[0] == 1.0
    assert currents[2] == 0.5
    assert currents[10] == 0.0


def test_reflex_decoder_rejects_wrong_shape() -> None:
    decoder = ReflexDecoder()
    with pytest.raises(ValueError, match="expected potentials shape"):
        decoder.decode(np.zeros(3))


def test_lif_and_direct_satisfy_model_backend() -> None:
    lif = InProcessLifBackend(build_reflex_connectome())
    direct = DirectBioSimBackend(n_neurons=4, model_id="vendor.fake")
    assert isinstance(lif, ModelBackend)
    assert isinstance(direct, ModelBackend)
    assert lif.n_neurons == 24
    assert direct.model_id == "vendor.fake"
