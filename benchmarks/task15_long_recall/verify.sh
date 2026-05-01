#!/bin/sh
# Verify task15_long_recall:
#   1. Source files must be untouched (read-only investigation task).
#   2. ANSWER.md must list each of the 15 unique keywords on the line
#      that matches the fact's number — proving the model actually
#      read each file (the keywords cannot be guessed from context).
#
# Why this task: 15 reads × ~770 chars produce ~12 KB of tool-result
# content in memory. Combined with system prompt + tool definitions,
# the per-call prompt exceeds the token-based compaction threshold
# (8192 × 0.6 = 4915 by default). The agent loop MUST compact mid-task
# to avoid context overflow on the writeup turn — this task verifies
# that compaction preserves enough information for the model to still
# emit all 15 correct keywords.
set -e

expected_keywords="
01:ZORTHAXIPLE_01
02:QUIBBLESNICK_02
03:WROBBLEFOX_03
04:MORTISLAW_04
05:BLATTERWICK_05
06:GREGGIANOSE_06
07:PURPLEFISH_07
08:KLAVERTRIDGE_08
09:SNOZZBERRY_09
10:GRIMTRACK_10
11:WOZZLEBUG_11
12:PLINKWORTH_12
13:SHEEPLEDORF_13
14:TWERPENBARK_14
15:JABBERFLAX_15
"

# 1. Anti-tampering: pin source SHA-256 so the model can't "fix" the
# answer by editing fact files to match what it remembered.
expected_shas="
fact_01.txt:f9358727484d658dce9e432ed4f68fe5fc5aec85af0c10d9d39defff075399cd
fact_02.txt:4d679cf2bb19b6733aec08fc94038b4a73b02dd2ad50f9803abacddbc542ef06
fact_03.txt:11bc4eec4fb71204738ac8532361cfc6cb7f7eb0be25b7ae81396407b4f47acd
fact_04.txt:2a1c5a1c82919f31266eb24e0bbf7bef418df22a041ed338dea68a8f50b7926f
fact_05.txt:85288b9d9033c81e138ae5b67c57405ce9ba2fe0e701aa594f92f90e8e746cce
fact_06.txt:80bed3ad1057a41811c72b78ca743cf7fa06b7ac594447bb011a4e85dd65a38d
fact_07.txt:ea3f16a280054d956fe71f80f184658caefca683de1769dcb8f6df09f72682a8
fact_08.txt:03b1d4e91f47cc331ebcbed215803640e6db858813eea319b2cefaae1b025fc6
fact_09.txt:8137dd32b68a4d7b0562cf78b5b2b2fe5b92cf1e2e939eb5f5bbb3f91b323ba0
fact_10.txt:b70422812f7b8a5333ff44ce90f07d73db3b24d9f8854c234b7ba5697ef324ba
fact_11.txt:2d8daab94fdf81fd3da60443faa1bc6c8e39850bcdcea1704f55cda90b4d803d
fact_12.txt:442d6729cdffdef3ecf5574c88a86ef307824090549f42e590e73a015348e9e2
fact_13.txt:03b772388874ddc5dbab70e50fe3dee76c5cdd70a390d9c2659d2fc3624ed193
fact_14.txt:c63adfb12f2a45c70873fa56252d22af7c9608c3b5a90db5eba57a874ce8124a
fact_15.txt:ed19e6adeab2b172af7a793fea0bdd6498f9bd6328c811e64b49584d93b60d00
"
echo "$expected_shas" | while IFS=: read -r file expected; do
    [ -z "$file" ] && continue
    actual=$(sha256sum "$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: source file '$file' was modified (read-only task)."
        echo "  expected SHA: $expected"
        echo "  actual   SHA: $actual"
        exit 1
    fi
done

# 2. ANSWER.md must exist.
if [ ! -f ANSWER.md ]; then
    echo "FAIL: ANSWER.md was not created"
    exit 1
fi

# 3. Each expected keyword must appear on a line that starts with the
# matching fact number. Whitespace and ': ' / ':' separators are
# tolerated; case-insensitive on the leading number, exact on the
# uppercase keyword.
python3 - <<'PY'
import re
import sys
import pathlib

answer = pathlib.Path("ANSWER.md").read_text()
expected = [
    ("01", "ZORTHAXIPLE_01"),
    ("02", "QUIBBLESNICK_02"),
    ("03", "WROBBLEFOX_03"),
    ("04", "MORTISLAW_04"),
    ("05", "BLATTERWICK_05"),
    ("06", "GREGGIANOSE_06"),
    ("07", "PURPLEFISH_07"),
    ("08", "KLAVERTRIDGE_08"),
    ("09", "SNOZZBERRY_09"),
    ("10", "GRIMTRACK_10"),
    ("11", "WOZZLEBUG_11"),
    ("12", "PLINKWORTH_12"),
    ("13", "SHEEPLEDORF_13"),
    ("14", "TWERPENBARK_14"),
    ("15", "JABBERFLAX_15"),
]

# Build a map number -> set of keywords on lines starting with that number
found = {}
for raw_line in answer.splitlines():
    line = raw_line.strip().lstrip("-*0123456789. ").strip("`*")
    # Match leading 1-2 digit number followed by ':' or '.'
    m = re.match(r"^\s*(\d{1,2})\s*[:.\-]\s*(.+)$", raw_line)
    if not m:
        continue
    num = m.group(1).zfill(2)
    rest = m.group(2)
    found.setdefault(num, []).append(rest)

missing = []
for n, kw in expected:
    candidates = found.get(n, [])
    if not any(kw in c.upper() for c in candidates):
        missing.append((n, kw, candidates))

if missing:
    print("FAIL: missing or incorrect keywords:")
    for n, kw, cs in missing:
        print(f"  - line {n}: expected '{kw}', got: {cs}")
    print("---- ANSWER.md ----")
    print(answer)
    sys.exit(1)

print(f"OK ({len(expected)} keywords listed correctly)")
PY
