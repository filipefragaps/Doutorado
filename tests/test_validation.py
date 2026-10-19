from harmonic_signatures.validation.checks import compare_h_max_across_sheets, validate_sheet


def test_validate_sheet_basic_counts(small_harmonicas_sheet):
    report = validate_sheet(small_harmonicas_sheet)

    assert report.n_rows == 3
    assert report.n_duplicate_rows == 0
    assert report.invalid_status_count == 1
    assert report.invalid_status_labels.to_dict() == {"DIODO 2": 1}
    assert set(report.label_counts.index) == {"DIODO 1", "DIODO 2"}
    assert (report.missing_pct == 0).all()


def test_validate_sheet_zero_pct(small_harmonicas_sheet):
    report = validate_sheet(small_harmonicas_sheet)
    assert report.zero_pct["H00 Va max [%]"] == 0
    assert "Va evt " not in report.zero_pct.index


def test_validate_sheet_no_status_column(small_interharmonicas_sheet):
    report = validate_sheet(small_interharmonicas_sheet)
    assert report.invalid_status_count == 0
    assert report.invalid_status_labels.empty


def test_compare_h_max_across_sheets_detects_shared_column(
    small_harmonicas_sheet, small_interharmonicas_sheet
):
    report = compare_h_max_across_sheets(
        small_harmonicas_sheet, small_interharmonicas_sheet, tolerance_seconds=30
    )

    assert report.n_compared_channels == 1
    assert "H01_Va_max" in report.per_channel_mismatch_pct.index
    assert report.mismatch_pct == 100.0
    assert report.max_abs_diff > 0
