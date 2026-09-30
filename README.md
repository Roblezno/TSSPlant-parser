# TSSPlant Result Parser

`tssplant_parser.py` is a dependency-free Python parser for `.res` files
written by **TSSPlant** and **TSSPlantS**. It converts the line-oriented output
into frozen, typed dataclasses while preserving query and prediction order.

The parser supports:

- Multiple query blocks, including duplicate query identifiers.
- TATA and TATA-less predictions.
- Queries with no predictions.
- Optional printed query sequences.
- TSSPlantS annotated gene starts and optional aggregate summaries.
- Strict structural validation with source filename and line-number errors.
- Flattening predictions into dictionaries for downstream analysis.

The core module uses only the Python standard library and requires Python
3.10 or newer.

## Quick Start

Run from this directory so that `tssplant_parser.py` is importable:

```python
from pathlib import Path

from tssplant_parser import parse_tssplant


result = parse_tssplant(Path("TSSPlant/TSSPlant_example.res"))

print(result.program)
print(result.input_file)

for query in result.queries:
    print(query.query_id, query.sequence_length)
    for prediction in query.predictions:
        print(
            prediction.promoter_class.value,
            prediction.tss_position,
            prediction.tss_score,
        )
```

`parse_tssplant` accepts a `str` path, a `pathlib.Path`, or an already-open
text handle. Caller-owned handles remain open:

```python
from tssplant_parser import parse_tssplant


with open("results.res", encoding="utf-8") as handle:
    result = parse_tssplant(handle)
```

## How Parsing Works

The parser uses an anchored, line-oriented state machine rather than fixed
column offsets or one expression over the whole file:

1. It validates the producer header and converts thresholds, interval, and
   policy fields to typed values.
2. It reads each `Query:` block, preserving the full header and optional
   printed sequence.
3. It recognizes either `Promoters not found` or a declared prediction count,
   then parses exactly that many TATA or TATA-less records.
4. For TSSPlantS, it optionally parses the summary, all nine distribution
   bins, and final selected-promoter totals.
5. It validates structural relationships that can be checked safely without
   reproducing the biological prediction algorithm.

Horizontal whitespace and LF or CRLF line endings are accepted. Numeric data
is parsed from labeled fields, not Fortran column positions.

## Data Model

`parse_tssplant` returns a `TSSPlantResult` containing:

- Global producer metadata and displayed thresholds.
- A tuple of `QueryResult` objects in source order.
- An optional structured `TSSPlantSummary` for TSSPlantS output.

Each `QueryResult` retains the complete FASTA defline from the `Query:` line.
Its `query_id` property conservatively returns the first non-whitespace token
after an optional leading `>`. No other metadata is inferred from the header.

Each `Prediction` has a `PromoterClass` value, TSS position and score, and
optional TATA-box fields. TATA-less predictions have `None` for both TATA-box
fields.

## Flattening Predictions

Use `prediction_rows` to obtain dictionaries suitable for CSV writing, JSON
processing, or construction of a DataFrame by an application that already
uses pandas:

```python
from tssplant_parser import parse_tssplant, prediction_rows


result = parse_tssplant("TSSPlant/TSSPlantS_example.res")
rows = prediction_rows(result)

for row in rows:
    print(row["query_id"], row["promoter_class"], row["tss_position"])
```

Every row contains:

- `query_header`
- `query_id`
- `query_length`
- `annotated_gene_start`
- `promoter_class`
- `tss_position`
- `tss_score`
- `tata_box_position`
- `tata_box_score`
- `relative_tss_position`

`relative_tss_position` is `tss_position - annotated_gene_start` when an
annotated start is present, otherwise `None`.

## Coordinate and Score Semantics

- TSS and TATA-box coordinates are 1-based and local to the supplied query
  orientation. They are not inherently genomic coordinates.
- `.res` files do not contain a strand field.
- `(TATA-)` means the TATA-less promoter class; it does not mean negative
  strand.
- A TSS score is a neural-network output difference, not a probability.
- Displayed thresholds are rounded producer metadata. The parser preserves
  them but does not reapply them to emitted predictions.
- TSSPlant has a known sorting defect that can associate a TATA box with the
  wrong TSS. The parser intentionally does not enforce TATA-to-TSS order,
  distance, or biological spacing.

## Validation and Errors

Strict mode is enabled by default. It validates required headers, numeric
fields, positive positions and sequence lengths, declared prediction counts,
printed sequence length, all nine summary bins, and unambiguous summary
totals. Fortran all-asterisk overflow fields are rejected because their values
cannot be recovered.

Malformed input raises `TSSPlantParseError`. Its `source` and `line_number`
attributes are available programmatically, and its message includes them when
known:

```python
from tssplant_parser import TSSPlantParseError, parse_tssplant


try:
    result = parse_tssplant("results.res")
except TSSPlantParseError as error:
    print(error.source, error.line_number, error)
```

Lenient mode is intentionally narrow:

```python
partial = parse_tssplant("partial.res", strict=False)
```

It permits unknown lines in the global header and a header-only partial file.
It still rejects malformed query metadata, prediction records, summaries, and
unexpected trailing data rather than fabricating or dropping values.

## Limitations

- Queries shorter than 251 nt are omitted by TSSPlant itself and cannot be
  recovered from a `.res` file.
- The parser does not infer strand, convert local coordinates to genomic
  coordinates, parse biological metadata from query headers, or reproduce the
  producer's prediction logic.
- The unsupported source-only `TSSPlantGW` variant is not parsed.

## Development

Run the test suite with:

```bash
python -m pytest -q
```

The tests parse both bundled `.res` examples directly and use inline synthetic
results for optional sequences, no-hit queries, malformed input, summaries,
line-numbered diagnostics, and flattening behavior. They do not read `.seq`
files or run TSSPlant.
