from video2tasks.eval.official_boundaries import score_boundary_recall


def test_score_boundary_recall_ignores_extra_predicted_boundaries() -> None:
    summary = score_boundary_recall(
        gt_boundaries=[100, 200],
        pred_boundaries=[10, 97, 102, 150, 198, 201, 250],
        tolerance_frames=5,
    )

    assert summary.hit_count == 2
    assert summary.miss_count == 0
    assert summary.recall == 1.0
    assert summary.pred_boundary_count == 7
    assert [match.hit for match in summary.matches] == [True, True]


def test_score_boundary_recall_uses_only_tight_boundary_hits() -> None:
    summary = score_boundary_recall(
        gt_boundaries=[100, 200, 300],
        pred_boundaries=[96, 209, 305],
        tolerance_frames=5,
    )

    assert summary.hit_count == 2
    assert summary.miss_count == 1
    assert summary.recall == 2 / 3
    assert summary.matches[0].delta_frames == -4
    assert summary.matches[0].hit is True
    assert summary.matches[1].delta_frames == 9
    assert summary.matches[1].hit is False
    assert summary.matches[2].delta_frames == 5
    assert summary.matches[2].hit is True
