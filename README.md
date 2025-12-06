# RL+Reflexion for Embodied AI in 3D Environments

**Improving vision-based reinforcement learning agents with VLM-based failure analysis and meta-control.**

---

## Overview

This project demonstrates that **vision-language models (VLMs) can improve RL agents** through post-hoc failure analysis and rule-based meta-control, achieving **55.6% relative improvement** on household robot tasks in the AI2-THOR 3D simulator.

**Main Result**: RL baseline (30% success) → RL+Reflexion (47% success) on ALFRED tasks.

### Motivation

Embodied AI agents trained with reinforcement learning often fail due to:
- **Limited exploration**: Offline RL agents struggle with systematic scene understanding
- **Repeated failures**: Agents get stuck in loops without learning from mistakes  
- **Missing task knowledge**: Agents don't leverage common-sense priors (e.g., "turn on lights before examining objects")

**Our Solution**: Use a large VLM (Qwen2-VL-7B) to analyze failed episodes and generate natural language rules, then implement a lightweight meta-controller that overrides the RL policy when beneficial.

---

## System Architecture

The final system consists of four components:

1. **Vision-Only RL Baseline (CQL)**
   - Offline RL trained on 33,943 expert transitions
   - ResNet-18 visual features (25,088-dim)
   - Conservative Q-Learning for robust value estimation
   - **Baseline performance**: 30.0% success (9/30 episodes)

2. **VLM-Based Failure Analysis**
   - Qwen2-VL-7B-Instruct (4-bit quantization)
   - Analyzes 21 failed episodes with frames + action logs
   - Generates 75 actionable rules (~3.6 per episode)

3. **Rule-Based Meta-Controller**
   - Parses VLM rules via keyword matching
   - Implements 4 action override heuristics:
     - 360° initial scan (triggered ~25/30 episodes)
     - Anti-repeat pickup (breaks failure loops)
     - LookDown when stuck
     - Late-episode toggle (for light tasks)

4. **Real 3D Evaluation**
   - AI2-THOR physics simulator
   - ALFRED household tasks (pick & place, examine in light)
   - 7 discrete actions, 100-step episodes

**Architecture Evolution**: The project initially aimed for EmBERT+PPO but was refocused to ResNet+CQL due to checkpoint compatibility and feature extraction constraints. This pivot enabled faster iteration and clearer empirical validation.

---

## Quick Start

### Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04)
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for VLM, optional for RL)
- **Python**: 3.9-3.13
- **Display**: X server for AI2-THOR visualization

### Installation

```bash
# Clone repository
git clone <repo-url>
cd RL-project

# Create virtual environment
python3 -m venv venv_thor
source venv_thor/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify AI2-THOR setup
export DISPLAY=:1.0
python3 -c "from ai2thor.controller import Controller; c = Controller(); print('✓ AI2-THOR ready')"
```

### Demo: 5-Episode Comparison

Run RL baseline vs RL+Reflexion on 5 episodes:

```bash
# Activate environment and set display
source venv_thor/bin/activate
export DISPLAY=:1.0

# Run RL baseline (5 episodes, ~3 minutes)
python3 rl/eval_in_3d_simulator.py \
    --agent rl \
    --num-episodes 5 \
    --split valid_seen

# Run RL+Reflexion (same 5 episodes, ~4 minutes)
python3 rl/eval_rl_reflexion_in_3d.py \
    --mode reflexion \
    --num-episodes 5 \
    --split valid_seen

# Compare results
cat data/logs/rl_3d_metrics.json | grep success_rate
cat data/logs/rl_reflexion_3d_metrics.json | grep success_rate
```

**Expected Output**:
- RL baseline: 1-2 successes out of 5
- RL+Reflexion: 2-3 successes out of 5
- Console logs show Reflexion overrides (e.g., "Step 1: override MoveAhead → RotateRight (360° scan)")

### Single Episode with Visualization

To see the agent in action with visual output:

```bash
# Run single episode with window visible (requires X server)
export DISPLAY=:0  # Use your actual display

python3 scripts/run_one_alfred_episode_3d.py \
    --split valid_seen \
    --episode-idx 0 \
    --max-steps 50
```

This will open a window showing the 3D environment and agent actions.

---

## Reproducing Full Results

### 1. IL3D-BC Training (Optional Baseline)

Train behavior cloning policy on 20 expert trajectories:

```bash
# Record BC dataset from expert replays (~10 minutes)
python3 scripts/record_il3d_bc_dataset.py

# Train BC policy (10 epochs, ~5 minutes)
python3 rl/train_il3d_bc.py

# Evaluate in 3D (expect 0% success)
python3 rl/eval_il3d_bc_in_simulator.py --num-episodes 20
```

### 2. RL Baseline Evaluation

Evaluate pretrained CQL policy:

```bash
# 30 episodes (~20 minutes)
python3 rl/eval_in_3d_simulator.py \
    --agent rl \
    --num-episodes 30 \
    --log-episodes \
    --log-dir data/logs/episodes_3d
```

**Expected**: 30% success (9/30 episodes)

### 3. VLM Rule Generation

Generate Reflexion rules from failed episodes:

```bash
# Requires GPU with 8+ GB VRAM for Qwen2-VL-7B
# Analyzes ~21 failures (~15 minutes)
python3 reflection/generate_rules_real_3d.py
```

**Output**: `data/rules/rule_database_real_3d.json` with 75 rules

### 4. RL+Reflexion Evaluation

Run full comparison:

```bash
# 30 episodes each for RL and RL+Reflexion (~40 minutes)
python3 rl/eval_rl_reflexion_in_3d.py \
    --mode both \
    --num-episodes 30
```

**Expected**: 47% success (14/30 episodes), +55.6% relative improvement

---

## Main Results

| Method | Success Rate | Avg Steps | Improvement |
|--------|--------------|-----------|-------------|
| IL3D-BC (Vision-Only) | 0.0% (0/20) | 80.0 | - |
| RL (CQL) | 30.0% (9/30) | 70.6 | baseline |
| **RL+Reflexion** | **46.7% (14/30)** | 57.1 | **+16.7%** (+55.6% rel) |

**Key Findings**:
- **Reflexion helped**: 7 episodes (RL failed → Reflexion succeeded)
- **Reflexion hurt**: 2 episodes (overhead on trivial tasks)
- **Most impactful heuristic**: 360° initial scan (triggered ~83% of episodes)

**Publication-Quality Figures**: See `docs/figures/` for all results visualizations.

---

## Pretrained Artifacts

The repository includes minimal pretrained artifacts for quick demo:

| Artifact | Path | Description |
|----------|------|-------------|
| CQL checkpoint | `models/offline_rl_cql/cql_policy.d3` | Main RL policy (100MB) |
| IL3D-BC checkpoint | `models/il3d_bc/bc_3d_policy_best.pth` | BC baseline (100MB) |
| Reflexion rules | `data/rules/rule_database_real_3d.json` | VLM-generated rules (17KB) |
| Sample frames | `data/sample_frames/valid_seen_0000/` | Example episode for VLM input |

---

## Regenerating Large Datasets

Large datasets are not included in the repo. Regenerate them as follows:

```bash
# 1. Offline RL dataset (~9GB, requires ALFRED data)
python3 rl/build_offline_dataset.py

# 2. IL3D-BC dataset (~65MB, requires AI2-THOR)
python3 scripts/record_il3d_bc_dataset.py

# 3. Episode frame logs (~24MB, requires AI2-THOR)
python3 rl/eval_in_3d_simulator.py --agent rl --num-episodes 30 --log-episodes
```

**Note**: ALFRED data (`data/json_feat_subset/`) must be downloaded separately. See [ALFRED documentation](https://github.com/askforalfred/alfred).

---

## Project Structure

```
RL-project/
├── rl/
│   ├── models/              # IL3D-BC, seq2seq definitions
│   ├── policies/            # RL, RL+Reflexion wrappers
│   ├── train_il3d_bc.py     # BC training
│   ├── eval_in_3d_simulator.py        # RL evaluation
│   └── eval_rl_reflexion_in_3d.py     # RL vs RL+Reflexion comparison
├── reflection/
│   ├── vlm_client_real.py            # Qwen2-VL-7B client
│   ├── reflexion_controller_3d.py    # Meta-controller
│   ├── episode_logging_3d.py         # Episode logging
│   ├── generate_rules_real_3d.py     # Rule generation
│   └── rule_db_real_3d.py            # Rule database helper
├── env/
│   └── wrappers/
│       └── alfred_sim_env_3d.py      # AI2-THOR wrapper
├── data/
│   ├── logs/                # Evaluation metrics (JSON)
│   ├── rules/               # VLM-generated rules
│   ├── sample_frames/       # Example episode frames for VLM input
│   └── splits/              # Dataset split definitions
├── models/
│   ├── offline_rl_cql/      # CQL checkpoint (99 MB)
│   └── il3d_bc/             # BC checkpoint (50 MB)
├── docs/
│   ├── figures/             # Publication figures
│   ├── tables/              # LaTeX/Markdown tables
│   └── *.md                 # Documentation
└── scripts/
    ├── generate_all_paper_figures.py  # Figure generation
    └── test_qwen2vl_real.py          # VLM sanity test
```

---

## Key Contributions

1. **First application of Reflexion to real 3D embodied AI** with a production VLM (Qwen2-VL-7B)
2. **Novel meta-controller design** translating VLM rules to action-level overrides
3. **Empirical validation** showing +55.6% relative improvement on ALFRED tasks
4. **Open-source implementation** with reproducible metrics and clear documentation

---

## Limitations and Future Work

**Current Limitations**:
- **Keyword-based heuristics**: Brittle, requires manual design
- **Small-scale evaluation**: 30 episodes (need 100+ for statistical power)
- **No direct policy conditioning**: Meta-controller can only override, not reshape the policy
- **Restricted task types**: Pick & place, examine in light (ALFRED has 6 more types)

**Promising Directions**:
1. **Semantic rule matching**: Use sentence embeddings instead of keywords
2. **Learned meta-policies**: Train neural network to map (state, rules) → override probability
3. **Active learning**: Selectively query VLM on high-uncertainty failures
4. **Memory integration**: Track what objects have been seen, actions tried
5. **Continuous control**: Extend to robotic manipulation beyond discrete actions
6. **Multi-task transfer**: Apply rules across task types

---

## Documentation

- **System Architecture**: `docs/system_architecture_narrative.md`
- **Results & Discussion**: `docs/results_and_discussion.md`
- **Training Guides**:
  - IL3D-BC: `docs/IL3D_BC_TRAINING.md`
  - Offline RL: `docs/RL_OFFLINE_TRAINING.md`
  - Reflexion: `docs/REFLEXION_MODULE.md`
- **Paper Summary**: `docs/PAPER_SUMMARY.md`

---

## Requirements

Key dependencies (see `requirements.txt` for full list):
- `ai2thor==5.0.0` - 3D simulator
- `torch>=2.0.0` - Deep learning
- `d3rlpy==2.5.0` - Offline RL (CQL)
- `transformers>=4.37.0` - Qwen2-VL
- `qwen-vl-utils` - VLM utilities
- `bitsandbytes` - 4-bit quantization

**GPU Memory**:
- RL training/evaluation: 2-4 GB
- VLM inference: 6-8 GB (4-bit quantization)
- Combined: 8-12 GB recommended

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{rl_reflexion_3d_2025,
  title={Improving Embodied AI with VLM-Based Reflexion: 
         From Offline RL to Meta-Control in 3D Environments},
  author={[Your Name]},
  year={2025},
  note={RL+Reflexion system achieving 55.6\% relative improvement 
        on ALFRED 3D tasks via Qwen2-VL-7B failure analysis}
}
```

---

## License

This project integrates multiple open-source components:
- **ALFRED**: MIT License
- **AI2-THOR**: Apache 2.0
- **d3rlpy**: MIT License
- **Transformers**: Apache 2.0

See individual component licenses for details.

---

## Acknowledgments

- **ALFRED dataset**: [Shridhar et al., CVPR 2020](https://askforalfred.com/)
- **AI2-THOR simulator**: [AllenAI](https://ai2thor.allenai.org/)
- **Conservative Q-Learning**: [Kumar et al., NeurIPS 2020](https://arxiv.org/abs/2006.04779)
- **Reflexion framework**: [Shinn et al., NeurIPS 2023](https://arxiv.org/abs/2303.11366)
- **Qwen2-VL**: [Alibaba Cloud](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

**Project Status**: ✅ Complete implementation with reproducible results (Nov 2025)
