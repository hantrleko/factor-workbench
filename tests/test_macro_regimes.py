"""The macro-state join does not use an estimate dated after as_of."""

from datetime import date

import pandas as pd

from factorlab.macro_regimes import align_states


def test_segment_stays_inside_the_selected_estimate():
    panel = pd.DataFrame(
        {
            "model_id": ["vix_weekly_level", "vix_weekly_level", "vix_weekly_level"],
            "estimation_end": ["2008-09-12", "2008-09-12", "2020-03-31"],
            "observation_date": ["2008-09-05", "2008-09-12", "2008-09-12"],
            "filtered_high": [0.1, 0.7, 0.99],
        }
    )
    aligned = align_states(panel, pd.Series(["2008-09-12", "2008-10-10"]), "vix_weekly_level", date(2008, 9, 15))
    assert aligned.iloc[0]["filtered_high"] == 0.7
    assert aligned.iloc[0]["high_state"] == 1
    assert pd.isna(aligned.iloc[1]["filtered_high"])