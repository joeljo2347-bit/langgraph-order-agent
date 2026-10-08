"""The per-call limits on model replies. Builds the client only; no model server is needed."""

import pytest
from langchain_core.messages import AIMessage

from order_agent import models


def test_default_model_gets_the_cap_and_timeout():
    m = models.load(models.DEFAULT)
    assert m.num_predict == models.MAX_TOKENS
    assert m.client_kwargs["timeout"] == models.TIMEOUT_S


def test_cap_is_configurable():
    m = models.load(models.DEFAULT, max_tokens=8192, timeout=300)
    assert (m.num_predict, m.client_kwargs["timeout"]) == (8192, 300)


@pytest.mark.parametrize("limits", [{"max_tokens": 64}, {"timeout": 0}])
def test_a_cap_too_small_for_normal_answers_is_refused(limits):
    with pytest.raises(ValueError):
        models.load(models.DEFAULT, **limits)


def test_default_cap_leaves_room_for_normal_answers():
    assert models.MAX_TOKENS >= 4 * models.MIN_TOKENS


def test_hit_cap_reads_either_stop_reason():
    assert models.hit_cap(AIMessage("cut", response_metadata={"done_reason": "length"}))
    assert models.hit_cap(AIMessage("cut", response_metadata={"stop_reason": "max_tokens"}))
    assert not models.hit_cap(AIMessage("done", response_metadata={"done_reason": "stop"}))
    assert not models.hit_cap(AIMessage("scripted"))
