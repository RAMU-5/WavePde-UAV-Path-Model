from experiments.exp12_fmm_baseline import check_metric_conversion


def test_exact_randers_conversion_matches_edge_time():
    assert check_metric_conversion() < 1e-12
