import os
from data_handling import Reader

def test_all_dates_have_10_stocks():
    # Compute correct path relative to this test file
    base_path = os.path.join(os.path.dirname(__file__), "../data_for_kit.csv")
    ltv_path = os.path.join(os.path.dirname(__file__), "../LTV_History.csv")
    vix_path = os.path.join(os.path.dirname(__file__), "../VIX_History.csv")

    reader = Reader(base_path=base_path, ltv_path=ltv_path, vix_path=vix_path)
    reader.read_data()
    reader.merge_all()
    df = reader.data

    counts = df.groupby("date")["sym_root"].nunique()
    assert counts.min() == 10, "Some dates have missing stocks"