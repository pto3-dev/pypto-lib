# Long-context QA validation status

The previous ALPHA/BRAVO two-turn examples were short-context checks.
They do not establish32K context extraction and remain labelled as short cases.

The long client uses four existing distinct archives, exactly32768 token IDs
per request, with specified evidence records near the beginning, middle and end.
The question requires three record objects, each with planned, completed,
budget, spent and dependency fields. Expected values are extracted from the
source archives. Each response must contain all15 fields with exact types and
values and finish naturally; server usage must confirm32768 input tokens.

Input preparation and device-response acceptance are separate. No long-context
model answer or pass rate is claimed until an actual run is saved. Current
card occupancy is recorded outside Git; foreign jobs are never stopped.

The real checkpoint tokenizer has confirmed four32768-token inputs. The selected
evidence positions are about0.14%,50%,99.4-99.7% of the retained context.
[Input audit](LONG_CONTEXT_INPUT_AUDIT.json) records hashes and measured offsets.
Device responses remain pending; the short examples are not substituted.
