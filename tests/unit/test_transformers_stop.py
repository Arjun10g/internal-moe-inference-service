from inference_service.engines.transformers import _BufferedStopFilter, _truncate_at_stop


def test_truncates_at_earliest_stop() -> None:
    assert _truncate_at_stop("answer END trailing STOP", ("STOP", "END")) == ("answer ", True)
    assert _truncate_at_stop("answer", ("END",)) == ("answer", False)


def test_stream_filter_removes_stop_across_chunks() -> None:
    stop_filter = _BufferedStopFilter(("<END>",))
    output = [
        stop_filter.push("answer<"),
        stop_filter.push("EN"),
        stop_filter.push("D>trailing"),
        stop_filter.finish(),
    ]
    assert "".join(output) == "answer"
    assert stop_filter.stopped


def test_stream_filter_flushes_tail_without_stop() -> None:
    stop_filter = _BufferedStopFilter(("<END>",))
    output = stop_filter.push("complete answer") + stop_filter.finish()
    assert output == "complete answer"
    assert not stop_filter.stopped
