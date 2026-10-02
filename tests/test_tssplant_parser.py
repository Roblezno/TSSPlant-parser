"""Tests for the dependency-free TSSPlant result parser."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from tssplant_parser import (
    PromoterClass,
    SelectionPolicy,
    TSSPlantParseError,
    parse_tssplant,
    prediction_rows,
)


ROOT = Path(__file__).resolve().parents[1]
TSSPLANT_EXAMPLE = ROOT / "TSSPlant" / "TSSPlant_example.res"
TSSPLANTS_EXAMPLE = ROOT / "TSSPlant" / "TSSPlantS_example.res"

TATA_LINE = (
    "TSS position: 201 TSS score = 1.2500 "
    "TATA-box position: 170 TATA-box score = 4.5000"
)
TATA_LESS_LINE = "(TATA-) TSS position: 250 TSS score = -0.7500"


def make_document(
    *query_blocks: str,
    program: str = "TSSPlant",
    policy: str = "promoter (TSS) closer to Gene start is taken",
    extra_header_line: str | None = None,
) -> str:
    """Build a concise synthetic result without requiring sequence fixtures."""

    lines = [
        f"Program {program}",
        "Search for RNA II promoters (TSSs)",
    ]
    if extra_header_line is not None:
        lines.append(extra_header_line)
    lines.extend(
        [
            "Input file with query sequence(s): example.seq",
            "Thresholds, for TATA promoters: -1.83",
            "TATA-less promoters: -1.78",
            "If in interval -/+ 300 nt around putative TSS both TATA and TATA-less promoters",
            f"are predicted, {policy}",
            "",
        ]
    )
    lines.extend(query_blocks)
    return "\n".join(lines) + "\n"


def make_query(
    body: str,
    *,
    header: str = ">query description",
    length: int = 300,
    annotated_start: int | None = None,
) -> str:
    """Build one query block around a supplied result body."""

    lines = [f"Query: {header}", f"Length of Query sequence: {length}"]
    if annotated_start is not None:
        lines.append(f"Annotated Gene Start Position: {annotated_start}")
    lines.extend([body, ""])
    return "\n".join(lines)


def make_valid_summary(*, total_tss: int = 1, first_row: str | None = None) -> str:
    """Build a complete one-gene TSSPlantS summary."""

    row = first_row or (
        "0-10 Left or Right ... 1 [1 TATA+ and 0 TATA-] "
        "... 100.000% [100.000% .. 0.000%]"
    )
    zero_rows = [
        f"{label} Left ... 0 [0 TATA+ and 0 TATA-] ... 0.000% [0.000% .. 0.000%]"
        for label in (
            "51-100",
            "101-200",
            "201-300",
            "301-400",
            "401-500",
            "501-600",
            "601/more",
        )
    ]
    lines = [
        "__________________________________________________",
        "Summary:",
        f"{total_tss} TSSs in 1 genes out of 1 genes",
        "TATA+ proms: 1",
        "TATA- proms: 0",
        "Prom density: 300 in 300 nt",
        "Distribution of Closest 1 TSSs relative to Gene Start (Around or Left):",
        row,
        "11-50 Left or Right ... 0 [0 TATA+ and 0 TATA-] ... 0.000% [0.000% .. 0.000%]",
        *zero_rows,
        "Totally: 1 [100.000%] TATA+ and 0 [0.000%] TATA- Proms",
    ]
    return "\n".join(lines)


def prediction_totals(path: Path) -> tuple[int, int, int, int]:
    """Return query, prediction, TATA, and TATA-less counts for a fixture."""

    result = parse_tssplant(path)
    predictions = [prediction for query in result.queries for prediction in query.predictions]
    tata_count = sum(item.promoter_class is PromoterClass.TATA for item in predictions)
    return len(result.queries), len(predictions), tata_count, len(predictions) - tata_count


def test_golden_prediction_totals() -> None:
    """Both bundled result files parse with the independently verified totals."""

    assert prediction_totals(TSSPLANT_EXAMPLE) == (17, 46, 9, 37)
    assert prediction_totals(TSSPLANTS_EXAMPLE) == (17, 46, 19, 27)


def test_golden_tssplant_details() -> None:
    """Ordinary output preserves headers, classes, and anomalous box positions."""

    result = parse_tssplant(TSSPLANT_EXAMPLE)
    expected_header = (
        "> PLPR0575 ..AC:S80135 ..OS:Arabidopsis thaliana ..GENE:HMG1 "
        "..PROD:enzyme 3-hidroxy-3-methylglutaryl-CoA reductase "
        "..[ -200: +51]  ..CDS:  +42 |Taxon: Dicot |Promoter: TATA-less| "
        "-2000:-1000, 2001: TSS"
    )

    assert result.program == "TSSPlant"
    assert result.queries[0].header == expected_header
    assert result.queries[0].query_id == "PLPR0575"
    assert result.summary is None
    assert all(query.annotated_gene_start is None for query in result.queries)
    assert result.queries[0].predictions[0].promoter_class is PromoterClass.TATA
    assert result.queries[0].predictions[1].promoter_class is PromoterClass.TATA_LESS

    anomalous = next(
        prediction
        for query in result.queries
        for prediction in query.predictions
        if prediction.tss_position == 635
    )
    assert anomalous.tata_box_position == 906


def test_golden_tssplants_summary() -> None:
    """TSSPlantS annotations and its complete summary are parsed structurally."""

    result = parse_tssplant(str(TSSPLANTS_EXAMPLE))
    summary = result.summary

    assert result.program == "TSSPlantS"
    assert all(query.annotated_gene_start == 1001 for query in result.queries)
    assert summary is not None
    assert (
        summary.total_tss,
        summary.genes_with_selected_tss,
        summary.genes_seen,
        summary.tata_total,
        summary.tata_less_total,
        summary.nucleotides_per_prediction,
        summary.total_nt,
    ) == (46, 17, 17, 19, 27, 406, 18_700)
    assert [item.label for item in summary.distribution] == [
        "0-10",
        "11-50",
        "51-100",
        "101-200",
        "201-300",
        "301-400",
        "401-500",
        "501-600",
        "601/more",
    ]


def test_tata_and_tata_less_predictions_preserve_source_order() -> None:
    """Multiple predictions map to typed classes without score-based inference."""

    block = make_query(f"2 promoter(s) predicted:\n{TATA_LINE}\n{TATA_LESS_LINE}")
    result = parse_tssplant(StringIO(make_document(block)))
    tata, tata_less = result.queries[0].predictions

    assert tata.promoter_class is PromoterClass.TATA
    assert (tata.tss_position, tata.tss_score, tata.tata_box_position, tata.tata_box_score) == (
        201,
        1.25,
        170,
        4.5,
    )
    assert tata_less.promoter_class is PromoterClass.TATA_LESS
    assert tata_less.tss_score == -0.75
    assert tata_less.tata_box_position is None
    assert tata_less.tata_box_score is None


def test_no_hit_query_has_empty_predictions() -> None:
    """The no-hit marker does not require a zero-count line."""

    result = parse_tssplant(StringIO(make_document(make_query("Promoters not found"))))

    assert result.queries[0].predictions == ()
    assert prediction_rows(result) == []


def test_empty_query_identifier_is_rejected() -> None:
    """A FASTA marker without an identifier cannot produce an output ID."""

    text = make_document(make_query("Promoters not found", header=">"))

    with pytest.raises(TSSPlantParseError, match="query identifier is empty"):
        parse_tssplant(StringIO(text))


def test_printed_sequence_extra_blanks_and_crlf_are_supported() -> None:
    """Sequence text is concatenated unchanged rather than normalized."""

    block = "\n".join(
        [
            "Query: >sequence-query",
            "",
            "Length of Query sequence: 8",
            "",
            "acgt",
            "NNcc",
            "",
            "",
            "Promoters not found",
            "",
        ]
    )
    text = make_document(block).replace("\n", "\r\n")

    result = parse_tssplant(StringIO(text))

    assert result.queries[0].sequence == "acgtNNcc"


@pytest.mark.parametrize(
    ("policy_text", "expected"),
    [
        ("TATA-less promoter is excluded", SelectionPolicy.PREFER_TATA),
        (
            "promoter (TSS) closer to Gene start is taken",
            SelectionPolicy.CLOSEST_TO_GENE_START,
        ),
    ],
)
def test_both_selection_policies(policy_text: str, expected: SelectionPolicy) -> None:
    """The producer's policy prose maps to stable enum values."""

    text = make_document(make_query("Promoters not found"), policy=policy_text)
    assert parse_tssplant(StringIO(text)).selection_policy is expected


def test_tssplants_without_summary_is_valid() -> None:
    """TSSPlantS output may end immediately after its query blocks."""

    block = make_query(
        "Promoters not found",
        annotated_start=275,
    )
    result = parse_tssplant(StringIO(make_document(block, program="TSSPlantS")))

    assert result.queries[0].annotated_gene_start == 275
    assert result.summary is None


def test_long_headers_and_duplicate_query_ids_are_preserved() -> None:
    """Headers remain opaque even when they resemble parser metadata."""

    first_header = (
        ">duplicate spaces: [a:b] Query: repeated "
        "Length of Query sequence: 999 Annotated Gene Start Position: 12"
    )
    second_header = ">duplicate another description"
    text = make_document(
        make_query("Promoters not found", header=first_header),
        make_query("Promoters not found", header=second_header),
    )

    result = parse_tssplant(StringIO(text))

    assert [query.header for query in result.queries] == [first_header, second_header]
    assert [query.query_id for query in result.queries] == ["duplicate", "duplicate"]


def test_horizontal_whitespace_variation_is_supported() -> None:
    """Labels are recognized despite harmless spaces-to-tabs changes."""

    text = make_document(make_query(f"1 promoter(s) predicted:\n{TATA_LINE}"))
    text = text.replace(
        "Thresholds, for TATA promoters: -1.83",
        "Thresholds,\tfor\tTATA\t\tpromoters:\t-1.83",
    ).replace("Query: >query description", "Query:\t>query description")

    result = parse_tssplant(StringIO(text))

    assert result.tata_threshold == -1.83
    assert result.queries[0].query_id == "query"


def test_caller_owned_handle_remains_open() -> None:
    """Parsing a supplied stream does not transfer ownership to the parser."""

    handle = StringIO(make_document(make_query("Promoters not found")))

    parse_tssplant(handle)

    assert not handle.closed


def test_prediction_rows_include_relative_coordinates() -> None:
    """Flattened rows contain plain class values and relative TSS positions."""

    block = make_query(
        f"1 promoter(s) predicted:\n{TATA_LINE}",
        header=">row-id details",
        annotated_start=225,
    )
    result = parse_tssplant(StringIO(make_document(block, program="TSSPlantS")))

    assert prediction_rows(result) == [
        {
            "query_header": ">row-id details",
            "query_id": "row-id",
            "query_length": 300,
            "annotated_gene_start": 225,
            "promoter_class": "TATA",
            "tss_position": 201,
            "tss_score": 1.25,
            "tata_box_position": 170,
            "tata_box_score": 4.5,
            "relative_tss_position": -24,
        }
    ]


def test_declared_prediction_count_mismatch_is_rejected() -> None:
    """A new query cannot silently terminate an incomplete prediction list."""

    first = make_query(f"2 promoter(s) predicted:\n{TATA_LINE}", header=">first")
    second = make_query("Promoters not found", header=">second")

    with pytest.raises(TSSPlantParseError, match="prediction count mismatch") as caught:
        parse_tssplant(StringIO(make_document(first, second)))

    assert caught.value.line_number is not None
    assert "expected 2 prediction records" in str(caught.value)


def test_malformed_prediction_has_source_and_line_diagnostic(tmp_path: Path) -> None:
    """Path inputs report their filename and the malformed record's line."""

    block = make_query("1 promoter(s) predicted:\nthis is not a prediction")
    path = tmp_path / "malformed.res"
    path.write_text(make_document(block), encoding="utf-8")

    with pytest.raises(TSSPlantParseError, match="malformed prediction record") as caught:
        parse_tssplant(path)

    assert caught.value.source == str(path)
    assert caught.value.line_number == 12
    assert f"{path}:12:" in str(caught.value)


def test_missing_query_length_is_rejected() -> None:
    """A result marker cannot substitute for required query metadata."""

    block = "Query: >missing-length\nPromoters not found\n"

    with pytest.raises(TSSPlantParseError, match="unexpected line") as caught:
        parse_tssplant(StringIO(make_document(block)))

    assert "Length of Query sequence" in str(caught.value)
    assert caught.value.line_number is not None


def test_fortran_numeric_overflow_is_targeted() -> None:
    """All-asterisk numeric fields fail as unrecoverable producer overflow."""

    overflow_line = "(TATA-) TSS position: 250 TSS score = ********"
    block = make_query(f"1 promoter(s) predicted:\n{overflow_line}")

    with pytest.raises(TSSPlantParseError, match="Fortran field overflow") as caught:
        parse_tssplant(StringIO(make_document(block)))

    assert caught.value.actual == "********"
    assert caught.value.line_number is not None


def test_printed_sequence_length_mismatch_is_rejected() -> None:
    """Printed sequence validation uses the reported query length."""

    block = make_query("acgt\nPromoters not found", length=5)

    with pytest.raises(TSSPlantParseError, match="printed sequence length"):
        parse_tssplant(StringIO(make_document(block)))


def test_inconsistent_summary_totals_are_rejected() -> None:
    """Summary totals must agree with parsed prediction records."""

    query = make_query(
        f"1 promoter(s) predicted:\n{TATA_LINE}",
        annotated_start=225,
    )
    text = make_document(
        f"{query}{make_valid_summary(total_tss=2)}",
        program="TSSPlantS",
    )

    with pytest.raises(TSSPlantParseError, match="summary TSS total") as caught:
        parse_tssplant(StringIO(text))

    assert caught.value.line_number is not None


def test_malformed_summary_row_is_rejected() -> None:
    """Each of the nine labeled distribution rows is mandatory."""

    query = make_query(
        f"1 promoter(s) predicted:\n{TATA_LINE}",
        annotated_start=225,
    )
    summary = make_valid_summary(first_row="0-10 malformed distribution row")
    text = make_document(f"{query}{summary}", program="TSSPlantS")

    with pytest.raises(TSSPlantParseError, match="unexpected line") as caught:
        parse_tssplant(StringIO(text))

    assert "0-10" in str(caught.value)


def test_unexpected_trailing_content_is_rejected() -> None:
    """Nonblank records after a complete summary are never ignored."""

    query = make_query(
        f"1 promoter(s) predicted:\n{TATA_LINE}",
        annotated_start=225,
    )
    text = make_document(f"{query}{make_valid_summary()}\ntrailing", program="TSSPlantS")

    with pytest.raises(TSSPlantParseError, match="after the summary"):
        parse_tssplant(StringIO(text))


def test_unknown_program_is_rejected() -> None:
    """Only the two supplied TSSPlant producers are supported."""

    text = make_document(make_query("Promoters not found"), program="TSSPlantGW")

    with pytest.raises(TSSPlantParseError, match="unknown program identifier"):
        parse_tssplant(StringIO(text))


def test_lenient_mode_allows_unknown_header_lines_and_partial_output() -> None:
    """Lenient recovery is limited to global metadata and header-only files."""

    text = make_document(extra_header_line="Build metadata: local installation")

    with pytest.raises(TSSPlantParseError):
        parse_tssplant(StringIO(text))

    result = parse_tssplant(StringIO(text), strict=False)

    assert result.input_file == "example.seq"
    assert result.queries == ()


def test_lenient_mode_does_not_discard_result_records() -> None:
    """Lenient mode still rejects prediction data outside a query block."""

    text = make_document("Promoters not found")

    with pytest.raises(TSSPlantParseError, match="outside a query block"):
        parse_tssplant(StringIO(text), strict=False)
