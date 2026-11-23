# Exact32K long-context QA result

An actual four-card run passed4/4 requests and60/60 factual fields. Each request contained exactly32768 token IDs, also confirmed by server usage. All responses finished naturally.

The selected records lie near0.14%,50%,99.4-99.7% of each retained context. Four distinct existing archives were used, without repeated padding. Each answer extracts planned, completed, budget, spent and dependency for three specified records.

TP4/PP1, MTP5, cards[4, 5, 6, 7]. Prefill and the drafter remain native vLLM; the complete target/verifier uses normally compiled PyPTO and prepared AICPU DIRECT. All four rank exit records match the validated images and record actual DIRECT submissions.

This is functional long-context acceptance, not a new10K performance baseline, a native/PTO long-output equivalence claim, or private Inner checkpoint certification.

| Request | Input tokens | Checked fields | Finish |
|---|---:|---:|---|
| 0 |32768|15/15|stop|
| 1 |32768|15/15|stop|
| 2 |32768|15/15|stop|
| 3 |32768|15/15|stop|

## Actual model responses

### Request0

```json
{
  "Z0-0000": {
    "planned": 328,
    "completed": 986,
    "budget": 1911,
    "spent": 2296,
    "dependency": "Z0-0000"
  },
  "Z0-0197": {
    "planned": 624,
    "completed": 959,
    "budget": 11297,
    "spent": 4368,
    "dependency": "Z0-0193"
  },
  "Z0-0391": {
    "planned": 43,
    "completed": 141,
    "budget": 11882,
    "spent": 301,
    "dependency": "Z0-0390"
  }
}
```

### Request1

```json
{
"Z1-0000": {
"planned": 698,
"completed": 440,
"budget": 7293,
"spent": 4886,
"dependency": "Z1-0000"
},
"Z1-0197": {
"planned": 690,
"completed": 438,
"budget": 1066,
"spent": 4830,
"dependency": "Z1-0191"
},
"Z1-0391": {
"planned": 301,
"completed": 287,
"budget": 5174,
"spent": 2107,
"dependency": "Z1-0389"
}
}
```

### Request2

```json
{
"Z2-0000": {
"planned": 29,
"completed": 224,
"budget": 4056,
"spent": 203,
"dependency": "Z2-0000"
},
"Z2-0197": {
"planned": 690,
"completed": 909,
"budget": 10452,
"spent": 4830,
"dependency": "Z2-0191"
},
"Z2-0391": {
"planned": 535,
"completed": 293,
"budget": 11154,
"spent": 3745,
"dependency": "Z2-0388"
}
}
```

### Request3

```json
{
"Z3-0000": {
"planned": 699,
"completed": 644,
"budget": 1443,
"spent": 4893,
"dependency": "Z3-0000"
},
"Z3-0196": {
"planned": 937,
"completed": 694,
"budget": 377,
"spent": 6559,
"dependency": "Z3-0191"
},
"Z3-0390": {
"planned": 943,
"completed": 875,
"budget": 7345,
"spent": 6601,
"dependency": "Z3-0382"
}
}
```

[Machine-readable responses and execution proof](LONG_CONTEXT_QA_RESULT.json).
[Input construction audit](LONG_CONTEXT_INPUT_AUDIT.json). Full token IDs, requests, responses, service logs and runtime exit records are retained in the local archival evidence.

The corpus is a publicly tracked synthetic project archive in
[GDD_ESCC/pypto](https://gitcode.com/GDD_ESCC/pypto), commit
`a62c3c176cc173e2ec84f35054a2e7bb403bc24d`. Its bytes match the committed file.
All60 published answer fields exactly match those public records. Each answer
is316-389 characters; raw32K input IDs and private checkpoint contents are not
included in the public result. Provenance is recorded in the result JSON.
