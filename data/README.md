# Data

Place the Sports Facts CSV here as `sports_original.csv` (3,132 rows, 1,559 athletes). It is
not included in this repository. The data are the Sports Facts of Nanda et al. (2023), "Fact
Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level", as used by Guo et
al. (2025), "Mechanistic Unlearning" (arXiv:2410.12949).

Expected columns: `athlete`, `sport` (baseball, basketball or football), `log_prob_one_shot`,
`num_athlete_tokens`, `sport_index`, `sport_token`, `prompt`, plus a leading unnamed index
column. Prompts have the form "Fact: Tiger Woods plays the sport of golf\nFact: <athlete> plays
the sport of".
