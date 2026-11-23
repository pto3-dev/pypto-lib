# Validated two-turn chat example

The executed source passed four conversations, two turns each: 8/8 requests and 32/32 fields.
This is a functional example using public mixed-W8A8 surrogate weights, not a performance or private-checkpoint certification.
The published client preserves the executed request/validation function bodies. Relocation is checked offline; this publication does not claim another NPU run.

| Case | Turn | Actual response |
|---|---:|---|
| ALPHA | 1 | `{"record_id":"ALPHA","planned":12,"completed":9,"budget":800,"spent":510}` |
| BRAVO | 1 | `{"record_id":"BRAVO","planned":20,"completed":7,"budget":1200,"spent":350}` |
| CHARLIE | 1 | `{"record_id":"CHARLIE","planned":35,"completed":16,"budget":2400,"spent":975}` |
| DELTA | 1 | `{"record_id":"DELTA","planned":9,"completed":4,"budget":600,"spent":125}` |
| ALPHA | 2 | `{"record_id":"ALPHA","remaining":3,"unspent":290}` |
| BRAVO | 2 | `{"record_id":"BRAVO","remaining":13,"unspent":850}` |
| CHARLIE | 2 | `{"record_id":"CHARLIE","remaining":19,"unspent":1425}` |
| DELTA | 2 | `{"record_id":"DELTA","remaining":5,"unspent":475}` |
