#!/usr/bin/env bash
set -euo pipefail

# Generated commands only; review data availability before running object_hand/background variants.
# Each command uses a single GPU selected round-robin from --gpus.

# run 0: task=place_dual_shoes setting=demo_clean seed=0 variant=vanilla gpu=1
bash train.sh place_dual_shoes demo_clean 50 0 1 32 200 100 latent_mode=pi3_full use_future_loss=true future_target_mode=pi3_full use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=demo_clean_vanilla_seed0 exp_name=place_dual_shoes_demo_clean_50_vanilla_seed0 logging.name=place_dual_shoes_demo_clean_50_vanilla_seed0

# run 1: task=place_dual_shoes setting=demo_clean seed=0 variant=dino_only gpu=2
bash train.sh place_dual_shoes demo_clean 50 0 2 32 200 100 latent_mode=dino_only use_future_loss=false future_target_mode=none use_pi3_features=false policy.use_pi3_features=false checkpoint_tag=demo_clean_dino_only_seed0 exp_name=place_dual_shoes_demo_clean_50_dino_only_seed0 logging.name=place_dual_shoes_demo_clean_50_dino_only_seed0

# run 2: task=place_dual_shoes setting=demo_clean seed=0 variant=no_future gpu=3
bash train.sh place_dual_shoes demo_clean 50 0 3 32 200 100 latent_mode=pi3_full use_future_loss=false future_target_mode=none use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=demo_clean_no_future_seed0 exp_name=place_dual_shoes_demo_clean_50_no_future_seed0 logging.name=place_dual_shoes_demo_clean_50_no_future_seed0

# run 3: task=place_dual_shoes setting=demo_clean seed=0 variant=pi3_pooled gpu=1
bash train.sh place_dual_shoes demo_clean 50 0 1 32 200 100 latent_mode=pi3_pooled use_future_loss=true future_target_mode=pi3_pooled use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=demo_clean_pi3_pooled_seed0 exp_name=place_dual_shoes_demo_clean_50_pi3_pooled_seed0 logging.name=place_dual_shoes_demo_clean_50_pi3_pooled_seed0

# run 4: task=place_dual_shoes setting=demo_clean seed=0 variant=pi3_compressed gpu=2
bash train.sh place_dual_shoes demo_clean 50 0 2 32 200 100 latent_mode=pi3_compressed use_future_loss=true future_target_mode=pi3_compressed use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=demo_clean_pi3_compressed_seed0 exp_name=place_dual_shoes_demo_clean_50_pi3_compressed_seed0 logging.name=place_dual_shoes_demo_clean_50_pi3_compressed_seed0

# run 5: task=place_dual_shoes setting=demo_clean seed=0 variant=pi3_random gpu=3
bash train.sh place_dual_shoes demo_clean 50 0 3 32 200 100 latent_mode=pi3_random_tokens use_future_loss=true future_target_mode=pi3_random_tokens use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=demo_clean_pi3_random_seed0 exp_name=place_dual_shoes_demo_clean_50_pi3_random_seed0 logging.name=place_dual_shoes_demo_clean_50_pi3_random_seed0

# run 6: task=place_dual_shoes setting=demo_clean seed=0 variant=pi3_dropout gpu=1
bash train.sh place_dual_shoes demo_clean 50 0 1 32 200 100 latent_mode=pi3_token_dropout use_future_loss=true future_target_mode=pi3_full use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=demo_clean_pi3_dropout_seed0 exp_name=place_dual_shoes_demo_clean_50_pi3_dropout_seed0 logging.name=place_dual_shoes_demo_clean_50_pi3_dropout_seed0
