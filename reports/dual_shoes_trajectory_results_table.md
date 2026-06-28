# Dual Shoes Public-50 Trajectory Results

GitHub issue note: official released 50-demo vanilla has been reported around 11/100.

These rows are public-50 mechanism evidence, not paper-main reproduction evidence.

| variant | training demos | seed | checkpoint | eval rollouts | success count | success rate | Wilson 95% CI | result path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | 50 | 0 | 200 | 50 | 4/50 | 0.080 | [0.032, 0.188] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean/seed_0/200/_result.txt | previous eval50 smoke evidence |
| dino_only | 50 | 0 | 200 | 50 | 8/50 | 0.160 | [0.083, 0.285] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| no_future | 50 | 0 | 200 | 50 | 1/50 | 0.020 | [0.004, 0.105] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_no_future_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_pooled | 50 | 0 | 200 | 50 | 6/50 | 0.120 | [0.056, 0.238] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_pooled_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_compressed | 50 | 0 | 200 | 50 | 2/50 | 0.040 | [0.011, 0.135] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_compressed_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_random | 50 | 0 | 200 | 50 | 0/50 | 0.000 | [0.000, 0.071] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_random_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_dropout | 50 | 0 | 200 | 50 | 4/50 | 0.080 | [0.032, 0.188] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_dropout_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_eef_region | 50 | 0 | 200 | 50 | 6/50 | 0.120 | [0.056, 0.238] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_eef_region_seed0/seed_0/200/_result.txt | EEF proxy, not true object-hand |
| pi3_non_eef_region | 50 | 0 | 200 | 50 | 5/50 | 0.100 | [0.043, 0.214] | results_eval50_full/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_non_eef_region_seed0/seed_0/200/_result.txt | EEF proxy complement |
| vanilla | 50 | 0 | 200 | 100 | 8/100 | 0.080 | [0.041, 0.150] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean/seed_0/200/_result.txt | previous eval50 smoke evidence |
| dino_only | 50 | 0 | 200 | 100 | 24/100 | 0.240 | [0.167, 0.332] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| no_future | 50 | 0 | 200 | 100 | 8/100 | 0.080 | [0.041, 0.150] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_no_future_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_pooled | 50 | 0 | 200 | 100 | 16/100 | 0.160 | [0.101, 0.244] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_pooled_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_compressed | 50 | 0 | 200 | 100 | 9/100 | 0.090 | [0.048, 0.162] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_compressed_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_random | 50 | 0 | 200 | 100 | 6/100 | 0.060 | [0.028, 0.125] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_random_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_dropout | 50 | 0 | 200 | 100 | 8/100 | 0.080 | [0.041, 0.150] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_dropout_seed0/seed_0/200/_result.txt | previous eval50 smoke evidence |
| pi3_eef_region | 50 | 0 | 200 | 100 | 18/100 | 0.180 | [0.117, 0.267] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_eef_region_seed0/seed_0/200/_result.txt | EEF proxy, not true object-hand |
| pi3_non_eef_region | 50 | 0 | 200 | 100 | 8/100 | 0.080 | [0.041, 0.150] | results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_non_eef_region_seed0/seed_0/200/_result.txt | EEF proxy complement |
| delta_future | 50 | 0 | 200 | 100 | 8/100 | 0.080 | [0.041, 0.150] | results_dual_shoes_50demo_trajectory_eval100/place_dual_shoes/GAP/demo_clean/delta_future_seed0/seed_0/200/_result.txt | future_target_mode=pi3_delta |
| changed_token_future | 50 | 0 | 200 | 100 | 9/100 | 0.090 | [0.048, 0.162] | results_dual_shoes_50demo_trajectory_eval100/place_dual_shoes/GAP/demo_clean/changed_token_future_seed0/seed_0/200/_result.txt | future_target_mode=pi3_changed_tokens |
| delta_changed_token_future | 50 | 0 | 200 | 100 | pending | pending | pending | results_dual_shoes_50demo_trajectory_eval100/place_dual_shoes/GAP/demo_clean/delta_changed_token_future_seed0/seed_0/200/_result.txt | optional pi3_delta_changed_tokens |
