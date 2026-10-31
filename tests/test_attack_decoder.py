from control.attack_decoder import (
    AttackDecoder,
)


def test_one_activation_one_attack():
    decoder = AttackDecoder(
        zero_confirm_steps=5
    )

    first = decoder.update(7)

    assert first.attack is True
    assert first.armed is False
    assert first.event_count == 1

    for _ in range(20):
        result = decoder.update(5)
        assert result.attack is False
        assert result.event_count == 1


def test_rearm_after_five_zero_steps():
    decoder = AttackDecoder(
        zero_confirm_steps=5
    )

    decoder.update(7)

    for _ in range(4):
        result = decoder.update(0)
        assert result.armed is False

    result = decoder.update(0)

    assert result.armed is True


def test_second_activation_produces_second_attack():
    decoder = AttackDecoder(
        zero_confirm_steps=5
    )

    assert decoder.update(7).attack is True

    for _ in range(5):
        decoder.update(0)

    second = decoder.update(1)

    assert second.attack is True
    assert second.event_count == 2


def test_persistent_activity_does_not_repeat():
    decoder = AttackDecoder()

    events = []

    for spikes in [
        0,
        0,
        7,
        5,
        4,
        6,
        5,
        3,
    ]:
        events.append(
            decoder.update(spikes).attack
        )

    assert sum(events) == 1


def test_delayed_onset_after_stimulus_is_valid():
    decoder = AttackDecoder()

    # Represents the observed attack #5 case:
    # sugar episode itself can show MN9=0,
    # then MN9 activates immediately afterward.
    assert decoder.update(0).attack is False
    assert decoder.update(0).attack is False

    delayed = decoder.update(5)

    assert delayed.attack is True


def test_negative_spikes_rejected():
    decoder = AttackDecoder()

    try:
        decoder.update(-1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "negative spike count must fail"
        )
