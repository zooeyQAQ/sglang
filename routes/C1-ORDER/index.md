# C1-ORDER

| EXP | status | SHA | baseline | finding |
| --- | --- | --- | --- | --- |
| EXP-0001 | invalid | baa9befa38 | none | SM120 candidate indexer refused |
| EXP-0002 | fail | 6ae2f5c2ef | EXP-0001 | N511 9 output_ids; fork site unmeasured |
| EXP-0003 | pass | 1a71ff57bf | EXP-0002 | N511 L20 same set/score, different order and sparse_out; N8193 not observed (OOM) |
| EXP-0004 | pass | 54e84fa9f3 | EXP-0003 | TP8 no DP, Engram host table; N511 L20 order fork reproduced; N8193 L2 C2 order fork observed |
