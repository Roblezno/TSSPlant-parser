"""Parse TSSPlant and TSSPlantS ``.res`` result files.

Coordinates in these files are 1-based positions in the supplied query
orientation. The format contains no strand field: ``(TATA-)`` identifies a
TATA-less promoter, not a negative-strand prediction. TSS scores are neural
network output differences rather than probabilities.

Some TSSPlant releases can associate an incorrect TATA box with a prediction
after sorting. This parser therefore preserves reported positions and does not
infer or validate biological spacing between a TSS and its TATA box.

Example:
    >>> result = parse_tssplant("TSSPlant/TSSPlant_example.res")
    >>> positions = [
    ...     (query.query_id, prediction.tss_position)
    ...     for query in result.queries
    ...     for prediction in query.predictions
    ... ]
    >>> positions[0]
    ('PLPR0575', 1026)
    >>> rows = prediction_rows(result)
    >>> rows[0]["promoter_class"]
    'TATA'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NoReturn, TextIO

__all__ = [
    "DistributionBin",
    "Prediction",
    "PromoterClass",
    "QueryResult",
    "SelectionPolicy",
    "TSSPlantParseError",
    "TSSPlantResult",
    "TSSPlantSummary",
    "parse_tssplant",
    "prediction_rows",
]


class PromoterClass(str, Enum):
    """Promoter classes emitted by TSSPlant."""

    TATA = "TATA"
    TATA_LESS = "TATA-less"


class SelectionPolicy(str, Enum):
    """Policy used when both promoter classes occur in one interval."""

    PREFER_TATA = "prefer_tata"
    CLOSEST_TO_GENE_START = "closest_to_gene_start"


@dataclass(frozen=True)
class Prediction:
    """One promoter prediction in source-file order."""

    promoter_class: PromoterClass
    tss_position: int
    tss_score: float
    tata_box_position: int | None = None
    tata_box_score: float | None = None


@dataclass(frozen=True)
class QueryResult:
    """Predictions and optional printed sequence for one query."""

    header: str
    sequence_length: int
    annotated_gene_start: int | None
    sequence: str | None
    predictions: tuple[Prediction, ...]

    @property
    def query_id(self) -> str:
        """Return the first token after an optional leading FASTA marker."""

        header = self.header.lstrip()
        if header.startswith(">"):
            header = header[1:].lstrip()
        return header.split(maxsplit=1)[0] if header else ""


@dataclass(frozen=True)
class DistributionBin:
    """One relative-position bin in a TSSPlantS summary."""

    label: str
    count: int
    tata_count: int
    tata_less_count: int
    percent: float
    tata_percent: float
    tata_less_percent: float


@dataclass(frozen=True)
class TSSPlantSummary:
    """Structured aggregate statistics optionally emitted by TSSPlantS."""

    total_tss: int
    genes_with_selected_tss: int
    genes_seen: int
    tata_total: int
    tata_less_total: int
    nucleotides_per_prediction: int
    total_nt: int
    distribution: tuple[DistributionBin, ...]
    selected_tata_total: int
    selected_tata_percent: float
    selected_tata_less_total: int
    selected_tata_less_percent: float


@dataclass(frozen=True)
class TSSPlantResult:
    """Complete parsed contents of a TSSPlant result file."""

    program: str
    input_file: str
    tata_threshold: float
    tata_less_threshold: float
    filtering_interval: int
    selection_policy: SelectionPolicy
    queries: tuple[QueryResult, ...]
    summary: TSSPlantSummary | None


class TSSPlantParseError(ValueError):
    """Raised when a TSSPlant result file is malformed or inconsistent."""

    source: str | None
    line_number: int | None
    expected: str | None
    actual: str | None

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        line_number: int | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        """Initialize an error with optional source and line context."""

        self.source = source
        self.line_number = line_number
        self.expected = expected
        self.actual = actual

        details = message
        if expected is not None:
            details += f"; expected {expected}"
        if actual is not None:
            details += f"; got {actual!r}"

        if source is not None and line_number is not None:
            details = f"{source}:{line_number}: {details}"
        elif source is not None:
            details = f"{source}: {details}"
        elif line_number is not None:
            details = f"line {line_number}: {details}"

        super().__init__(details)


_PROGRAM_RE = re.compile(r"^[ \t]*Program[ \t]+(?P<program>\S+)[ \t]*$")
_PURPOSE_RE = re.compile(
    r"^[ \t]*Search[ \t]+for[ \t]+RNA[ \t]+II[ \t]+promoters[ \t]*"
    r"\([ \t]*TSSs[ \t]*\)[ \t]*$"
)
_INPUT_RE = re.compile(
    r"^[ \t]*Input[ \t]+file[ \t]+with[ \t]+query[ \t]+sequence\(s\):"
    r"[ \t]*(?P<input>.*?)[ \t]*$"
)
_TATA_THRESHOLD_RE = re.compile(
    r"^[ \t]*Thresholds,[ \t]*for[ \t]+TATA[ \t]+promoters:"
    r"[ \t]*(?P<value>\S+)[ \t]*$"
)
_TATA_LESS_THRESHOLD_RE = re.compile(
    r"^[ \t]*TATA-less[ \t]+promoters:[ \t]*(?P<value>\S+)[ \t]*$"
)
_INTERVAL_RE = re.compile(
    r"^[ \t]*If[ \t]+in[ \t]+interval[ \t]+-/\+[ \t]*(?P<value>\S+)"
    r"[ \t]+nt[ \t]+around[ \t]+putative[ \t]+TSS[ \t]+both[ \t]+TATA"
    r"[ \t]+and[ \t]+TATA-less[ \t]+promoters[ \t]*$"
)
_PREFER_TATA_RE = re.compile(
    r"^[ \t]*are[ \t]+predicted,[ \t]+TATA-less[ \t]+promoter"
    r"[ \t]+is[ \t]+excluded[ \t]*$"
)
_CLOSEST_RE = re.compile(
    r"^[ \t]*are[ \t]+predicted,[ \t]+promoter[ \t]*\([ \t]*TSS[ \t]*\)"
    r"[ \t]+closer[ \t]+to[ \t]+Gene[ \t]+start[ \t]+is[ \t]+taken[ \t]*$"
)
_QUERY_RE = re.compile(r"^[ \t]*Query:[ \t]*(?P<header>.*)$")
_QUERY_LENGTH_RE = re.compile(
    r"^[ \t]*Length[ \t]+of[ \t]+Query[ \t]+sequence:"
    r"[ \t]*(?P<value>\S+)[ \t]*$"
)
_GENE_START_RE = re.compile(
    r"^[ \t]*Annotated[ \t]+Gene[ \t]+Start[ \t]+Position:"
    r"[ \t]*(?P<value>\S+)[ \t]*$"
)
_NO_HIT_RE = re.compile(r"^[ \t]*Promoters[ \t]+not[ \t]+found[ \t]*$")
_COUNT_RE = re.compile(
    r"^[ \t]*(?P<value>\S+)[ \t]+promoter\(s\)[ \t]+predicted:[ \t]*$"
)
_TATA_PREDICTION_RE = re.compile(
    r"^[ \t]*TSS[ \t]+position:[ \t]*(?P<tss_position>\S+)"
    r"[ \t]+TSS[ \t]+score[ \t]*=[ \t]*(?P<tss_score>\S+)"
    r"[ \t]+TATA-box[ \t]+position:[ \t]*(?P<tata_position>\S+)"
    r"[ \t]+TATA-box[ \t]+score[ \t]*=[ \t]*(?P<tata_score>\S+)[ \t]*$"
)
_TATA_LESS_PREDICTION_RE = re.compile(
    r"^[ \t]*\([ \t]*TATA-[ \t]*\)[ \t]+TSS[ \t]+position:"
    r"[ \t]*(?P<tss_position>\S+)[ \t]+TSS[ \t]+score[ \t]*="
    r"[ \t]*(?P<tss_score>\S+)[ \t]*$"
)
_SEQUENCE_RE = re.compile(r"^[A-Za-z]+$")
_SUMMARY_SEPARATOR_RE = re.compile(r"^[ \t]*_{50}[ \t]*$")
_SUMMARY_TITLE_RE = re.compile(r"^[ \t]*Summary:[ \t]*$")
_SUMMARY_TOTAL_RE = re.compile(
    r"^[ \t]*(?P<total>\S+)[ \t]+TSSs[ \t]+in[ \t]+(?P<selected>\S+)"
    r"[ \t]+genes[ \t]+out[ \t]+of[ \t]+(?P<seen>\S+)[ \t]+genes[ \t]*$"
)
_SUMMARY_TATA_RE = re.compile(
    r"^[ \t]*TATA\+[ \t]+proms:[ \t]*(?P<value>\S+)[ \t]*$"
)
_SUMMARY_TATA_LESS_RE = re.compile(
    r"^[ \t]*TATA-[ \t]+proms:[ \t]*(?P<value>\S+)[ \t]*$"
)
_SUMMARY_DENSITY_RE = re.compile(
    r"^[ \t]*Prom[ \t]+density:[ \t]*(?P<density>\S+)"
    r"[ \t]+in[ \t]+(?P<total_nt>\S+)[ \t]+nt[ \t]*$"
)
_DISTRIBUTION_TITLE_RE = re.compile(
    r"^[ \t]*Distribution[ \t]+of[ \t]+Closest[ \t]+(?P<count>\S+)"
    r"[ \t]+TSSs[ \t]+relative[ \t]+to[ \t]+Gene[ \t]+Start"
    r"[ \t]*\([ \t]*Around[ \t]+or[ \t]+Left[ \t]*\):[ \t]*$"
)
_DISTRIBUTION_ROW_RE = re.compile(
    r"^[ \t]*(?P<label>\S+)[ \t]+(?:Left[ \t]+or[ \t]+Right|Left)"
    r"[ \t]+\.\.\.[ \t]*(?P<count>\S+)[ \t]*"
    r"\[[ \t]*(?P<tata_count>\S+)[ \t]+TATA\+[ \t]+and[ \t]+"
    r"(?P<tata_less_count>\S+)[ \t]+TATA-[ \t]*\][ \t]*"
    r"\.\.\.[ \t]*(?P<percent>[^ \t%]+)%[ \t]*"
    r"\[[ \t]*(?P<tata_percent>[^ \t%]+)%[ \t]*\.\.[ \t]*"
    r"(?P<tata_less_percent>[^ \t%]+)%[ \t]*\][ \t]*$"
)
_SUMMARY_SELECTED_RE = re.compile(
    r"^[ \t]*Totally:[ \t]*(?P<tata_count>\S+)[ \t]*"
    r"\[[ \t]*(?P<tata_percent>[^ \t%]+)%[ \t]*\][ \t]+TATA\+"
    r"[ \t]+and[ \t]+(?P<tata_less_count>\S+)[ \t]*"
    r"\[[ \t]*(?P<tata_less_percent>[^ \t%]+)%[ \t]*\]"
    r"[ \t]+TATA-[ \t]+Proms[ \t]*$"
)
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_OVERFLOW_RE = re.compile(r"^\*+$")

_EXPECTED_BINS = (
    "0-10",
    "11-50",
    "51-100",
    "101-200",
    "201-300",
    "301-400",
    "401-500",
    "501-600",
    "601/more",
)

_GLOBAL_PATTERNS = (
    _PROGRAM_RE,
    _PURPOSE_RE,
    _INPUT_RE,
    _TATA_THRESHOLD_RE,
    _TATA_LESS_THRESHOLD_RE,
    _INTERVAL_RE,
    _PREFER_TATA_RE,
    _CLOSEST_RE,
)


class _Parser:
    """Line-oriented parser retaining source positions for diagnostics."""

    def __init__(self, text: str, source: str | None, strict: bool) -> None:
        self._lines = text.splitlines()
        self._source = source
        self._strict = strict

    def parse(self) -> TSSPlantResult:
        """Parse all header, query, and optional summary records."""

        (
            index,
            program,
            input_file,
            tata_threshold,
            tata_less_threshold,
            filtering_interval,
            selection_policy,
        ) = self._parse_global_header()

        queries: list[QueryResult] = []
        summary: TSSPlantSummary | None = None

        while True:
            index = self._skip_blank_lines(index)
            if index >= len(self._lines):
                break

            line = self._lines[index]
            query_match = _QUERY_RE.fullmatch(line)
            if query_match is not None:
                query, index = self._parse_query(index, query_match, program)
                queries.append(query)
                continue

            if _SUMMARY_SEPARATOR_RE.fullmatch(line) is not None:
                if program != "TSSPlantS":
                    self._raise(
                        index,
                        "a summary is only valid for TSSPlantS output",
                        expected="a query block or end of file",
                    )
                summary, index = self._parse_summary(index, queries)
                break

            if self._is_result_record(line):
                self._raise(
                    index,
                    "result record appears outside a query block",
                    expected="a 'Query:' header",
                )
            if not self._strict and not queries:
                index += 1
                continue
            self._raise(
                index,
                "unexpected content after the global header",
                expected="a 'Query:' header, summary separator, or end of file",
            )

        trailing_index = self._skip_blank_lines(index)
        if trailing_index < len(self._lines):
            self._raise(
                trailing_index,
                "unexpected nonblank content after the summary",
                expected="end of file",
            )

        if not queries and self._strict:
            self._raise(
                len(self._lines),
                "no query blocks were found",
                expected="at least one 'Query:' block",
                actual="<end of file>",
            )

        return TSSPlantResult(
            program=program,
            input_file=input_file,
            tata_threshold=tata_threshold,
            tata_less_threshold=tata_less_threshold,
            filtering_interval=filtering_interval,
            selection_policy=selection_policy,
            queries=tuple(queries),
            summary=summary,
        )

    def _parse_global_header(
        self,
    ) -> tuple[int, str, str, float, float, int, SelectionPolicy]:
        index = 0
        program_match, index = self._expect_header(
            index,
            _PROGRAM_RE,
            "a 'Program TSSPlant' or 'Program TSSPlantS' line",
        )
        program = program_match.group("program")
        if program not in {"TSSPlant", "TSSPlantS"}:
            self._raise(
                index - 1,
                f"unknown program identifier {program!r}",
                expected="TSSPlant or TSSPlantS",
            )

        _, index = self._expect_header(index, _PURPOSE_RE, "the TSSPlant purpose line")
        input_match, index = self._expect_header(index, _INPUT_RE, "the input filename line")
        input_file = input_match.group("input")
        if not input_file:
            self._raise(index - 1, "the displayed input filename is empty", expected="a filename")

        tata_match, index = self._expect_header(
            index,
            _TATA_THRESHOLD_RE,
            "the TATA promoter threshold line",
        )
        tata_threshold = self._parse_float(
            tata_match.group("value"),
            index - 1,
            "TATA threshold",
        )

        tata_less_match, index = self._expect_header(
            index,
            _TATA_LESS_THRESHOLD_RE,
            "the TATA-less promoter threshold line",
        )
        tata_less_threshold = self._parse_float(
            tata_less_match.group("value"),
            index - 1,
            "TATA-less threshold",
        )

        interval_match, index = self._expect_header(
            index,
            _INTERVAL_RE,
            "the promoter filtering interval line",
        )
        filtering_interval = self._parse_int(
            interval_match.group("value"),
            index - 1,
            "filtering interval",
        )

        policy_index = index
        prefer_match, next_index = self._match_header(index, _PREFER_TATA_RE)
        if prefer_match is not None:
            selection_policy = SelectionPolicy.PREFER_TATA
            index = next_index
        else:
            closest_match, next_index = self._match_header(index, _CLOSEST_RE)
            if closest_match is None:
                self._raise(
                    policy_index,
                    "missing or unknown selection policy",
                    expected=(
                        "'TATA-less promoter is excluded' or "
                        "'promoter (TSS) closer to Gene start is taken'"
                    ),
                )
            selection_policy = SelectionPolicy.CLOSEST_TO_GENE_START
            index = next_index

        return (
            index,
            program,
            input_file,
            tata_threshold,
            tata_less_threshold,
            filtering_interval,
            selection_policy,
        )

    def _parse_query(
        self,
        index: int,
        query_match: re.Match[str],
        program: str,
    ) -> tuple[QueryResult, int]:
        header = query_match.group("header")
        if not header.strip():
            self._raise(index, "query header is empty", expected="a complete FASTA defline")
        index += 1

        length_match, index = self._expect_nonblank(
            index,
            _QUERY_LENGTH_RE,
            "a 'Length of Query sequence:' line",
        )
        sequence_length = self._parse_int(
            length_match.group("value"),
            index - 1,
            "query sequence length",
        )
        if sequence_length <= 0:
            self._raise(
                index - 1,
                "query sequence length must be positive",
                expected="an integer greater than zero",
                actual=length_match.group("value"),
            )

        annotated_gene_start: int | None = None
        if program == "TSSPlantS":
            gene_start_match, index = self._expect_nonblank(
                index,
                _GENE_START_RE,
                "an 'Annotated Gene Start Position:' line",
            )
            annotated_gene_start = self._parse_int(
                gene_start_match.group("value"),
                index - 1,
                "annotated gene start",
            )
            if annotated_gene_start <= 0:
                self._raise(
                    index - 1,
                    "annotated gene start must be positive",
                    expected="an integer greater than zero",
                    actual=gene_start_match.group("value"),
                )

        sequence_lines: list[str] = []
        while True:
            index = self._skip_blank_lines(index)
            if index >= len(self._lines):
                self._raise(
                    index,
                    "query block ends before its result marker",
                    expected="'Promoters not found' or a prediction count",
                    actual="<end of file>",
                )

            line = self._lines[index]
            no_hit_match = _NO_HIT_RE.fullmatch(line)
            count_match = _COUNT_RE.fullmatch(line)
            if no_hit_match is not None or count_match is not None:
                break

            if _QUERY_RE.fullmatch(line) is not None or _SUMMARY_SEPARATOR_RE.fullmatch(line) is not None:
                self._raise(
                    index,
                    "query block has no result marker",
                    expected="'Promoters not found' or a prediction count",
                )
            if self._is_prediction(line):
                self._raise(
                    index,
                    "prediction appears before its declared count",
                    expected="a '<integer> promoter(s) predicted:' line",
                )
            if _SEQUENCE_RE.fullmatch(line) is None:
                self._raise(
                    index,
                    "malformed optional query sequence line",
                    expected="alphabetic sequence text or a result marker",
                )
            sequence_lines.append(line)
            index += 1

        sequence = "".join(sequence_lines) if sequence_lines else None
        if sequence is not None and len(sequence) != sequence_length:
            self._raise(
                index,
                "printed sequence length does not match the reported query length",
                expected=f"{sequence_length} sequence characters",
                actual=f"{len(sequence)} sequence characters",
            )

        if no_hit_match is not None:
            return (
                QueryResult(
                    header=header,
                    sequence_length=sequence_length,
                    annotated_gene_start=annotated_gene_start,
                    sequence=sequence,
                    predictions=(),
                ),
                index + 1,
            )

        assert count_match is not None
        prediction_count = self._parse_int(
            count_match.group("value"),
            index,
            "declared prediction count",
        )
        if prediction_count <= 0:
            self._raise(
                index,
                "a prediction count line must declare at least one prediction",
                expected="an integer greater than zero",
                actual=count_match.group("value"),
            )
        index += 1

        predictions: list[Prediction] = []
        while len(predictions) < prediction_count:
            index = self._skip_blank_lines(index)
            if index >= len(self._lines):
                self._raise_count_mismatch(index, prediction_count, len(predictions), header)

            line = self._lines[index]
            if _QUERY_RE.fullmatch(line) is not None or _SUMMARY_SEPARATOR_RE.fullmatch(line) is not None:
                self._raise_count_mismatch(index, prediction_count, len(predictions), header)

            prediction = self._parse_prediction(index)
            predictions.append(prediction)
            index += 1

        return (
            QueryResult(
                header=header,
                sequence_length=sequence_length,
                annotated_gene_start=annotated_gene_start,
                sequence=sequence,
                predictions=tuple(predictions),
            ),
            index,
        )

    def _parse_prediction(self, index: int) -> Prediction:
        line = self._lines[index]
        tata_less_match = _TATA_LESS_PREDICTION_RE.fullmatch(line)
        if tata_less_match is not None:
            tss_position = self._parse_positive_position(
                tata_less_match.group("tss_position"),
                index,
                "TSS position",
            )
            tss_score = self._parse_float(
                tata_less_match.group("tss_score"),
                index,
                "TSS score",
            )
            return Prediction(
                promoter_class=PromoterClass.TATA_LESS,
                tss_position=tss_position,
                tss_score=tss_score,
            )

        tata_match = _TATA_PREDICTION_RE.fullmatch(line)
        if tata_match is not None:
            tss_position = self._parse_positive_position(
                tata_match.group("tss_position"),
                index,
                "TSS position",
            )
            tss_score = self._parse_float(
                tata_match.group("tss_score"),
                index,
                "TSS score",
            )
            tata_position = self._parse_positive_position(
                tata_match.group("tata_position"),
                index,
                "TATA-box position",
            )
            tata_score = self._parse_float(
                tata_match.group("tata_score"),
                index,
                "TATA-box score",
            )
            return Prediction(
                promoter_class=PromoterClass.TATA,
                tss_position=tss_position,
                tss_score=tss_score,
                tata_box_position=tata_position,
                tata_box_score=tata_score,
            )

        self._raise(
            index,
            "malformed prediction record",
            expected="a TATA or '(TATA-)' prediction line",
        )

    def _parse_summary(
        self,
        index: int,
        queries: list[QueryResult],
    ) -> tuple[TSSPlantSummary, int]:
        index += 1
        _, index = self._expect_nonblank(index, _SUMMARY_TITLE_RE, "a 'Summary:' line")

        totals_match, index = self._expect_nonblank(
            index,
            _SUMMARY_TOTAL_RE,
            "the summary TSS and gene totals line",
        )
        totals_line = index - 1
        total_tss = self._parse_nonnegative_int(totals_match.group("total"), totals_line, "summary TSS total")
        genes_with_selected_tss = self._parse_nonnegative_int(
            totals_match.group("selected"),
            totals_line,
            "genes with a selected TSS",
        )
        genes_seen = self._parse_nonnegative_int(totals_match.group("seen"), totals_line, "genes seen")

        tata_match, index = self._expect_nonblank(
            index,
            _SUMMARY_TATA_RE,
            "the summary TATA+ total line",
        )
        tata_line = index - 1
        tata_total = self._parse_nonnegative_int(tata_match.group("value"), tata_line, "summary TATA+ total")

        tata_less_match, index = self._expect_nonblank(
            index,
            _SUMMARY_TATA_LESS_RE,
            "the summary TATA- total line",
        )
        tata_less_line = index - 1
        tata_less_total = self._parse_nonnegative_int(
            tata_less_match.group("value"),
            tata_less_line,
            "summary TATA- total",
        )

        density_match, index = self._expect_nonblank(
            index,
            _SUMMARY_DENSITY_RE,
            "the promoter density line",
        )
        density_line = index - 1
        nucleotides_per_prediction = self._parse_nonnegative_int(
            density_match.group("density"),
            density_line,
            "nucleotides per prediction",
        )
        total_nt = self._parse_nonnegative_int(
            density_match.group("total_nt"),
            density_line,
            "summary nucleotide total",
        )

        distribution_match, index = self._expect_nonblank(
            index,
            _DISTRIBUTION_TITLE_RE,
            "the distribution heading",
        )
        distribution_title_line = index - 1
        distribution_total = self._parse_nonnegative_int(
            distribution_match.group("count"),
            distribution_title_line,
            "distribution TSS total",
        )

        distribution: list[DistributionBin] = []
        for expected_label in _EXPECTED_BINS:
            row_match, index = self._expect_nonblank(
                index,
                _DISTRIBUTION_ROW_RE,
                f"the {expected_label!r} distribution row",
            )
            row_line = index - 1
            label = row_match.group("label")
            if label != expected_label:
                self._raise(
                    row_line,
                    "distribution bins are missing or out of order",
                    expected=expected_label,
                    actual=label,
                )

            count = self._parse_nonnegative_int(row_match.group("count"), row_line, f"{label} count")
            tata_count = self._parse_nonnegative_int(
                row_match.group("tata_count"),
                row_line,
                f"{label} TATA count",
            )
            tata_less_count = self._parse_nonnegative_int(
                row_match.group("tata_less_count"),
                row_line,
                f"{label} TATA-less count",
            )
            percent = self._parse_float(row_match.group("percent"), row_line, f"{label} percentage")
            tata_percent = self._parse_float(
                row_match.group("tata_percent"),
                row_line,
                f"{label} TATA percentage",
            )
            tata_less_percent = self._parse_float(
                row_match.group("tata_less_percent"),
                row_line,
                f"{label} TATA-less percentage",
            )

            if tata_count + tata_less_count != count:
                self._raise(
                    row_line,
                    f"distribution bin {label!r} has inconsistent class counts",
                    expected=f"{count} combined TATA and TATA-less predictions",
                    actual=str(tata_count + tata_less_count),
                )

            distribution.append(
                DistributionBin(
                    label=label,
                    count=count,
                    tata_count=tata_count,
                    tata_less_count=tata_less_count,
                    percent=percent,
                    tata_percent=tata_percent,
                    tata_less_percent=tata_less_percent,
                )
            )

        selected_match, index = self._expect_nonblank(
            index,
            _SUMMARY_SELECTED_RE,
            "the final selected-promoter totals line",
        )
        selected_line = index - 1
        selected_tata_total = self._parse_nonnegative_int(
            selected_match.group("tata_count"),
            selected_line,
            "selected TATA total",
        )
        selected_tata_percent = self._parse_float(
            selected_match.group("tata_percent"),
            selected_line,
            "selected TATA percentage",
        )
        selected_tata_less_total = self._parse_nonnegative_int(
            selected_match.group("tata_less_count"),
            selected_line,
            "selected TATA-less total",
        )
        selected_tata_less_percent = self._parse_float(
            selected_match.group("tata_less_percent"),
            selected_line,
            "selected TATA-less percentage",
        )

        parsed_predictions = [
            prediction
            for query in queries
            for prediction in query.predictions
        ]
        parsed_tata_total = sum(
            prediction.promoter_class is PromoterClass.TATA
            for prediction in parsed_predictions
        )
        parsed_tata_less_total = len(parsed_predictions) - parsed_tata_total
        distribution_count = sum(item.count for item in distribution)
        distribution_tata_count = sum(item.tata_count for item in distribution)
        distribution_tata_less_count = sum(item.tata_less_count for item in distribution)

        self._validate_equal(
            total_tss,
            len(parsed_predictions),
            totals_line,
            "summary TSS total does not match parsed predictions",
        )
        self._validate_equal(
            tata_total,
            parsed_tata_total,
            tata_line,
            "summary TATA+ total does not match parsed predictions",
        )
        self._validate_equal(
            tata_less_total,
            parsed_tata_less_total,
            tata_less_line,
            "summary TATA- total does not match parsed predictions",
        )
        self._validate_equal(
            tata_total + tata_less_total,
            total_tss,
            tata_less_line,
            "summary class totals do not add up to the summary TSS total",
        )
        self._validate_equal(
            distribution_total,
            genes_with_selected_tss,
            distribution_title_line,
            "distribution heading does not match the selected-gene total",
        )
        self._validate_equal(
            distribution_count,
            genes_with_selected_tss,
            selected_line,
            "distribution-bin counts do not match the selected-gene total",
        )
        self._validate_equal(
            selected_tata_total,
            distribution_tata_count,
            selected_line,
            "selected TATA total does not match the distribution bins",
        )
        self._validate_equal(
            selected_tata_less_total,
            distribution_tata_less_count,
            selected_line,
            "selected TATA-less total does not match the distribution bins",
        )
        self._validate_equal(
            selected_tata_total + selected_tata_less_total,
            genes_with_selected_tss,
            selected_line,
            "selected promoter totals do not match the selected-gene total",
        )
        if genes_with_selected_tss > genes_seen:
            self._raise(
                totals_line,
                "selected-gene total exceeds the number of genes seen",
                expected=f"at most {genes_seen}",
                actual=str(genes_with_selected_tss),
            )

        return (
            TSSPlantSummary(
                total_tss=total_tss,
                genes_with_selected_tss=genes_with_selected_tss,
                genes_seen=genes_seen,
                tata_total=tata_total,
                tata_less_total=tata_less_total,
                nucleotides_per_prediction=nucleotides_per_prediction,
                total_nt=total_nt,
                distribution=tuple(distribution),
                selected_tata_total=selected_tata_total,
                selected_tata_percent=selected_tata_percent,
                selected_tata_less_total=selected_tata_less_total,
                selected_tata_less_percent=selected_tata_less_percent,
            ),
            index,
        )

    def _expect_header(
        self,
        index: int,
        pattern: re.Pattern[str],
        expected: str,
    ) -> tuple[re.Match[str], int]:
        if self._strict:
            return self._expect_at(index, pattern, expected)

        while index < len(self._lines):
            match = pattern.fullmatch(self._lines[index])
            if match is not None:
                return match, index + 1
            if self._is_global_structure(self._lines[index]) or self._is_block_start(self._lines[index]):
                break
            index += 1
        return self._expect_at(index, pattern, expected)

    def _match_header(
        self,
        index: int,
        pattern: re.Pattern[str],
    ) -> tuple[re.Match[str] | None, int]:
        if self._strict:
            if index >= len(self._lines):
                return None, index
            match = pattern.fullmatch(self._lines[index])
            return match, index + 1 if match is not None else index

        while index < len(self._lines):
            match = pattern.fullmatch(self._lines[index])
            if match is not None:
                return match, index + 1
            if self._is_global_structure(self._lines[index]) or self._is_block_start(self._lines[index]):
                return None, index
            index += 1
        return None, index

    def _expect_nonblank(
        self,
        index: int,
        pattern: re.Pattern[str],
        expected: str,
    ) -> tuple[re.Match[str], int]:
        return self._expect_at(self._skip_blank_lines(index), pattern, expected)

    def _expect_at(
        self,
        index: int,
        pattern: re.Pattern[str],
        expected: str,
    ) -> tuple[re.Match[str], int]:
        if index >= len(self._lines):
            self._raise(
                index,
                "unexpected end of file",
                expected=expected,
                actual="<end of file>",
            )
        match = pattern.fullmatch(self._lines[index])
        if match is None:
            self._raise(index, "unexpected line", expected=expected)
        return match, index + 1

    def _parse_positive_position(self, token: str, index: int, field: str) -> int:
        value = self._parse_int(token, index, field)
        if value <= 0:
            self._raise(
                index,
                f"{field} must be positive",
                expected="an integer greater than zero",
                actual=token,
            )
        return value

    def _parse_nonnegative_int(self, token: str, index: int, field: str) -> int:
        value = self._parse_int(token, index, field)
        if value < 0:
            self._raise(
                index,
                f"{field} cannot be negative",
                expected="a nonnegative integer",
                actual=token,
            )
        return value

    def _parse_int(self, token: str, index: int, field: str) -> int:
        self._reject_overflow(token, index, field)
        if _INTEGER_RE.fullmatch(token) is None:
            self._raise(
                index,
                f"malformed integer for {field}",
                expected="a base-10 integer",
                actual=token,
            )
        return int(token)

    def _parse_float(self, token: str, index: int, field: str) -> float:
        self._reject_overflow(token, index, field)
        if _FLOAT_RE.fullmatch(token) is None:
            self._raise(
                index,
                f"malformed decimal value for {field}",
                expected="a signed or unsigned decimal number",
                actual=token,
            )
        return float(token)

    def _reject_overflow(self, token: str, index: int, field: str) -> None:
        if _OVERFLOW_RE.fullmatch(token) is not None:
            self._raise(
                index,
                f"Fortran field overflow for {field}; the original value is unrecoverable",
                expected="a numeric value",
                actual=token,
            )

    def _validate_equal(
        self,
        actual: int,
        expected: int,
        index: int,
        message: str,
    ) -> None:
        if actual != expected:
            self._raise(
                index,
                message,
                expected=str(expected),
                actual=str(actual),
            )

    def _raise_count_mismatch(
        self,
        index: int,
        declared: int,
        parsed: int,
        header: str,
    ) -> None:
        self._raise(
            index,
            f"prediction count mismatch for query {header!r}",
            expected=f"{declared} prediction records",
            actual=f"{parsed} prediction records before the next block or end of file",
        )

    def _skip_blank_lines(self, index: int) -> int:
        while index < len(self._lines) and not self._lines[index].strip():
            index += 1
        return index

    @staticmethod
    def _is_global_structure(line: str) -> bool:
        return any(pattern.fullmatch(line) is not None for pattern in _GLOBAL_PATTERNS)

    @staticmethod
    def _is_block_start(line: str) -> bool:
        return (
            _QUERY_RE.fullmatch(line) is not None
            or _SUMMARY_SEPARATOR_RE.fullmatch(line) is not None
        )

    @staticmethod
    def _is_prediction(line: str) -> bool:
        return (
            _TATA_PREDICTION_RE.fullmatch(line) is not None
            or _TATA_LESS_PREDICTION_RE.fullmatch(line) is not None
        )

    @classmethod
    def _is_result_record(cls, line: str) -> bool:
        return (
            cls._is_prediction(line)
            or _COUNT_RE.fullmatch(line) is not None
            or _NO_HIT_RE.fullmatch(line) is not None
        )

    def _raise(
        self,
        index: int,
        message: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
    ) -> NoReturn:
        if actual is None:
            actual = self._lines[index] if index < len(self._lines) else "<end of file>"
        raise TSSPlantParseError(
            message,
            source=self._source,
            line_number=index + 1,
            expected=expected,
            actual=actual,
        )


def parse_tssplant(
    source: str | Path | TextIO,
    *,
    strict: bool = True,
) -> TSSPlantResult:
    """Parse a TSSPlant or TSSPlantS result file.

    Args:
        source: Filesystem path or caller-owned open text handle.
        strict: Whether to reject unknown global-header lines and files with no
            query blocks. All query, prediction, and summary records remain
            structurally validated when this is false.

    Returns:
        The complete typed result in source-file order.

    Raises:
        TSSPlantParseError: If required data is malformed, missing, or
            structurally inconsistent.
        TypeError: If ``source`` is not a path or readable text handle.
    """

    source_name: str | None
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8", newline=None) as handle:
            text = handle.read()
        source_name = str(source)
    else:
        read = getattr(source, "read", None)
        if read is None:
            raise TypeError("source must be a path or readable text handle")
        text = read()
        if not isinstance(text, str):
            raise TypeError("source must provide text, not bytes")
        handle_name = getattr(source, "name", None)
        source_name = str(handle_name) if isinstance(handle_name, (str, Path)) else None

    return _Parser(text, source_name, strict).parse()


def prediction_rows(result: TSSPlantResult) -> list[dict[str, object]]:
    """Flatten all predictions into analysis-ready row dictionaries.

    The relative TSS position is ``tss_position - annotated_gene_start`` when
    the TSSPlantS output provides an annotated start. No strand inference or
    coordinate conversion is performed.

    Args:
        result: Parsed TSSPlant result.

    Returns:
        One dictionary per prediction, preserving query and prediction order.
    """

    rows: list[dict[str, object]] = []
    for query in result.queries:
        for prediction in query.predictions:
            relative_position = (
                prediction.tss_position - query.annotated_gene_start
                if query.annotated_gene_start is not None
                else None
            )
            rows.append(
                {
                    "query_header": query.header,
                    "query_id": query.query_id,
                    "query_length": query.sequence_length,
                    "annotated_gene_start": query.annotated_gene_start,
                    "promoter_class": prediction.promoter_class.value,
                    "tss_position": prediction.tss_position,
                    "tss_score": prediction.tss_score,
                    "tata_box_position": prediction.tata_box_position,
                    "tata_box_score": prediction.tata_box_score,
                    "relative_tss_position": relative_position,
                }
            )
    return rows
