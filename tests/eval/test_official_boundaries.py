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


def test_score_boundary_recall_counts_hits_within_inclusive_tolerance() -> None:
    summary = score_boundary_recall(
        gt_boundaries=[100, 200, 300],
        pred_boundaries=[96, 209, 305],
        tolerance_frames=5,
    )

    assert summary.hit_count == 2
    assert summary.miss_count == 1
    assert summary.recall == 2 / 3
    assert [match.matched_pred_frame for match in summary.matches] == [96, 209, 305]
    assert [match.delta_frames for match in summary.matches] == [-4, 9, 5]
    assert summary.matches[0].hit is True
    assert summary.matches[1].hit is False
    assert summary.matches[2].hit is True


def test_score_boundary_recall_allows_each_prediction_to_hit_only_one_gt() -> None:
    summary = score_boundary_recall(
        gt_boundaries=[100, 104],
        pred_boundaries=[102],
        tolerance_frames=3,
    )

    assert summary.hit_count == 1
    assert summary.miss_count == 1
    assert summary.recall == 0.5
    assert [match.hit for match in summary.matches] == [True, False]
    assert summary.matches[0].matched_pred_frame == 102


def test_score_boundary_recall_finds_maximum_one_to_one_hits_with_unordered_inputs() -> None:
    summary = score_boundary_recall(
        gt_boundaries=[104, 100],
        pred_boundaries=[102, 97],
        tolerance_frames=3,
    )

    assert summary.hit_count == 2
    assert summary.miss_count == 0
    assert summary.recall == 1.0
    assert [match.hit for match in summary.matches] == [True, True]
    # Matches are reported in the original GT order.
    assert [match.gt_frame for match in summary.matches] == [104, 100]
    assert [match.matched_pred_frame for match in summary.matches] == [102, 97]
    assert [match.delta_frames for match in summary.matches] == [-2, -3]


def test_score_boundary_recall_is_zero_when_no_ground_truth_boundaries() -> None:
    summary = score_boundary_recall(
        gt_boundaries=[],
        pred_boundaries=[100, 200],
        tolerance_frames=5,
    )

    assert summary.gt_boundary_count == 0
    assert summary.hit_count == 0
    assert summary.miss_count == 0
    assert summary.recall == 0.0
