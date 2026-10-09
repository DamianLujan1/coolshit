import polars as pl

from gridiron.recap import write_recap


def _recap(model_hits, market_hits):
    n = len(model_hits)
    graded = pl.DataFrame({
        "week": [4] * n,
        "model_hit": pl.Series(model_hits, dtype=pl.Int8),
        "market_hit": pl.Series(market_hits, dtype=pl.Int8),
        "pick_prob": [0.6] * n,
        "pick": ["DAL"] * n,
    })
    return write_recap(2026, 5, pl.DataFrame(), graded, {})


def test_market_comparison_uses_only_common_games():
    # The extra model win has no market benchmark. It cannot establish a win
    # against the market, which tied the model on the two comparable games.
    text = _recap([1, 0, 1], [None, 1, 0])
    assert "model went 2-1 on 3 decisive graded games" in text
    assert "model went 1-1 and the moneyline favorite went 1-1" in text
    assert "we tied the market" in text
    assert "Closing moneylines were unavailable for 1" in text
    assert "we beat the market" not in text


def test_no_market_data_does_not_become_losses():
    text = _recap([1, 0], [None, None])
    assert "no market comparison yet" in text
    assert "moneyline favorite went" not in text


def test_ties_are_pushes_and_recap_does_not_claim_publication():
    text = _recap([1, 0, None], [1, 0, None])
    assert "model went 1-1 on 2 decisive graded games" in text
    assert "1 tied game(s), counted as pushes" in text
    assert "the 2 decisive games with closing moneylines" in text
    assert "committed to a public git log" not in text
