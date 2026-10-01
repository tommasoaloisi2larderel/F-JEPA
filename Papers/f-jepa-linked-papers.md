# F-JEPA linked papers

Deep-search index created on 2026-09-30.

Scope used here:

- the F-JEPA paper itself;
- papers cited by F-JEPA's arXiv manuscript;
- papers that are directly adjacent by title/abstract to Flow-JEPA, JEPA world models, JEPA control, or flow-matching latent dynamics;
- broader JEPA variants found across arXiv/OpenReview/project pages.

## Core F-JEPA

| Paper | Link | Quick summary |
| --- | --- | --- |
| Flow-JEPA: Robust Latent Dynamics for JEPA World Models via Flow Matching | [arXiv](https://arxiv.org/abs/2608.29029) / [code](https://github.com/HuoYanchen/Flow-JEPA) | Introduces F-JEPA, a JEPA world model that replaces deterministic autoregressive latent dynamics with conditional flow matching over complete future latent trajectories. Reports stronger latent planning under clean and visually perturbed observations. |

## Direct JEPA world-model lineage

| Paper | Link | Quick summary |
| --- | --- | --- |
| A Path Towards Autonomous Machine Intelligence | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | LeCun's position paper introducing JEPA as a prediction-in-latent-space route toward world models, planning, and abstraction. |
| Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | [arXiv](https://arxiv.org/abs/2301.08243) | I-JEPA learns image representations by predicting masked target-block embeddings from context-block embeddings without pixel reconstruction or contrastive negatives. |
| MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features | [arXiv](https://arxiv.org/abs/2307.12698) | Learns motion and content features jointly, tying optical-flow-style motion learning to JEPA representation learning. |
| Revisiting Feature Prediction for Learning Visual Representations from Video | [arXiv](https://arxiv.org/abs/2404.08471) | V-JEPA trains video representations through latent feature prediction alone, giving F-JEPA an important video/self-supervised predecessor. |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | [arXiv](https://arxiv.org/abs/2506.09985) | Extends V-JEPA toward larger video understanding and planning-capable predictive representations. |
| LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics | [arXiv](https://arxiv.org/abs/2511.08544) | Introduces the SIGReg-style Gaussian regularization family used by LeWorldModel and F-JEPA to reduce representation collapse risk. |
| LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [arXiv](https://arxiv.org/abs/2603.19312) / [project](https://le-wm.github.io/) | F-JEPA's main baseline. Learns an end-to-end pixel-to-latent JEPA world model with next-embedding prediction and Gaussian regularization, then plans with CEM/MPC. |
| DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning | [arXiv](https://arxiv.org/abs/2411.04983) / [PMLR](https://proceedings.mlr.press/v267/zhou25t.html) | Learns action-conditioned dynamics over frozen DINOv2 visual features, enabling zero-shot planning toward goal observations. |
| Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models | [arXiv](https://arxiv.org/abs/2502.14819) / [project](https://latent-planning.github.io/) | Introduces PLDM-style planning with learned latent dynamics from reward-free offline trajectories. |
| Stress-Testing Offline Reward-Free Reinforcement Learning: A Case for Planning with Latent Dynamics Models | [OpenReview PDF](https://openreview.net/pdf?id=jON7H6A9UU) | Earlier/workshop version around latent-dynamics planning under reward-free offline data and generalization stress tests. |
| VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models | [arXiv](https://arxiv.org/abs/2601.14354) | Generalizes JEPA toward probabilistic future-latent prediction and uncertainty-aware planning. |
| VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model | [arXiv](https://arxiv.org/abs/2602.10098) / [project](https://ginwind.github.io/VLA-JEPA/) | Uses JEPA-style latent future-state prediction for VLA policy pretraining, emphasizing leakage-free prediction and robustness to nuisance appearance. |

## Closely related JEPA control and planning papers

| Paper | Link | Quick summary |
| --- | --- | --- |
| FF-JEPA: Long-Horizon Planning in World Models with Latent Planners | [arXiv](https://arxiv.org/abs/2606.09311) | Adds a latent subgoal planner on top of JEPA world models to reduce long-horizon CEM planning difficulty. |
| Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control | [arXiv](https://arxiv.org/abs/2607.25337) | Teaches JEPA representations a directed temporal-distance cost mined from offline logs, narrowing the gap between training loss and plan-time ranking. |
| Qantara: Bridge-Flow Training for Multi-Paradigm JEPA Control | [arXiv](https://arxiv.org/abs/2607.04978) | Combines Brownian-bridge state interpolation with action-axis flow matching so one JEPA checkpoint can support planning, behavior cloning, and inverse dynamics. |
| Calibrated Predictive Safety for Heterogeneous Robots: An Action-Conditioned JEPA Framework with Model-Based Safety Shields | [arXiv](https://arxiv.org/abs/2608.17496) | Uses an action-conditioned JEPA rollout model to score progress and risk, with deterministic safety shields enforcing robot constraints. |
| Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization | [arXiv](https://arxiv.org/abs/2609.10464) | Extends LeWorldModel-style JEPA dynamics with law/parameter conditioning to test zero-shot physics generalization. |
| Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations | [arXiv](https://arxiv.org/abs/2608.29434) | Moves action-conditioned JEPA planning from images to point-cloud observations, testing whether latent planning survives sparse geometric inputs. |
| JEPA-VLA: Video Predictive Embedding is Needed for VLA Models | [arXiv](https://arxiv.org/abs/2602.11832) | Shows that predictive video embeddings, especially V-JEPA-style ones, improve sample efficiency and generalization in VLA manipulation models. |

## Flow matching and stochastic latent dynamics

| Paper | Link | Quick summary |
| --- | --- | --- |
| Flow Matching for Generative Modeling | [arXiv](https://arxiv.org/abs/2210.02747) | Core conditional-flow-matching foundation used by F-JEPA to learn vector fields from source trajectories to future latent targets. |
| Rectified Flow: A Marginal Preserving Approach to Optimal Transport | [arXiv](https://arxiv.org/abs/2209.14577) | Early rectified-flow formulation for transporting one distribution into another. |
| Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | [arXiv](https://arxiv.org/abs/2209.03003) | Develops rectified flows with straighter transport paths and faster generation. |
| Flow Matching in Feature Space for Stochastic World Modeling | [arXiv](https://arxiv.org/abs/2606.29059) | FlowWM performs flow matching directly in pretrained feature space to model stochastic futures. Very close conceptually to F-JEPA's latent-dynamics motivation. |
| Action-to-Action Flow Matching | [arXiv](https://arxiv.org/abs/2602.07322) | Uses structured previous-action sources rather than pure Gaussian noise for fast flow-based robot action generation. |
| Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching | [arXiv](https://arxiv.org/abs/2602.05951) | Studies how learnable condition-dependent flow sources can improve conditional flow matching, relevant to F-JEPA's Gaussian source design choices. |
| pi0: A Vision-Language-Action Flow Model for General Robot Control | [arXiv](https://arxiv.org/abs/2410.24164) | Uses flow-based action generation for general robot control, connecting flow objectives with embodied action prediction. |
| Wan: Open and Advanced Large-Scale Video Generative Models | [arXiv](https://arxiv.org/abs/2503.20314) | Large-scale video generation work cited by F-JEPA as an example of modern conditional generation. |

## Latent world-model and planning baselines

| Paper | Link | Quick summary |
| --- | --- | --- |
| World Models | [arXiv](https://arxiv.org/abs/1803.10122) | Classic learned latent world-model setup for imagined rollouts and control. |
| Learning Latent Dynamics for Planning from Pixels | [PMLR](https://proceedings.mlr.press/v97/hafner19a.html) | PlaNet learns latent dynamics from image observations and plans with model-predictive control. |
| Dream to Control: Learning Behaviors by Latent Imagination | [arXiv](https://arxiv.org/abs/1912.01603) | Dreamer learns policies by actor-critic training inside a learned latent world model. |
| Mastering Atari with Discrete World Models | [arXiv](https://arxiv.org/abs/2010.02193) | DreamerV2 scales world-model RL to Atari with discrete latent variables. |
| Mastering Diverse Domains through World Models | [arXiv](https://arxiv.org/abs/2301.04104) | DreamerV3 demonstrates a fixed-hyperparameter world-model agent across many domains. |
| Training Agents Inside of Scalable World Models | [arXiv](https://arxiv.org/abs/2509.24527) | Dreamer 4 scales imagination training and offline world-model learning to much larger, complex environments. |
| Transformers are Sample-Efficient World Models | [arXiv](https://arxiv.org/abs/2209.00588) | IRIS models visual tokens autoregressively as a sample-efficient Atari world model. |
| Diffusion for World Modeling: Visual Details Matter in Atari | [arXiv](https://arxiv.org/abs/2405.12399) | DIAMOND uses diffusion as a world model, arguing that preserving visual detail can matter for control. |
| Frozen Forecasting: A Unified Evaluation | [arXiv](https://arxiv.org/abs/2507.13942) | Evaluates future prediction in frozen/pretrained feature spaces, relevant to non-reconstructive world modeling. |
| TD-MPC: Temporal Difference Learning for Model Predictive Control | [arXiv](https://arxiv.org/abs/2203.04955) | Combines learned latent dynamics, value prediction, and model-predictive control. |
| TD-MPC2: Scalable, Robust World Models for Continuous Control | [arXiv](https://arxiv.org/abs/2310.16828) | Scales TD-MPC-style implicit world models over many continuous-control tasks. |

## JEPA variants across modalities and domains

| Paper | Link | Quick summary |
| --- | --- | --- |
| A-JEPA: Joint-Embedding Predictive Architecture Can Listen | [arXiv](https://arxiv.org/abs/2311.15830) | Extends JEPA-style latent prediction to audio spectrograms with time-frequency-aware masking. |
| Stem-JEPA: A Joint-Embedding Predictive Architecture for Musical Stem Compatibility Estimation | [arXiv](https://arxiv.org/abs/2408.02514) | Learns musical stem compatibility by predicting embeddings of compatible stems from mix-context embeddings. |
| Music-JEPA: Learning a World Model of Sound from Action | [arXiv](https://arxiv.org/abs/2607.22000) | Frames piano audio as state and pianoroll as action, using JEPA to learn a sound world model. |
| MIDI-RAE-JEPA: Hierarchical Representation Learning and Generation for Symbolic Music | [arXiv](https://arxiv.org/abs/2607.14537) | Combines LeJEPA/SIGReg-style representation learning with equivariance objectives and flow-matching generation for symbolic music. |
| MoRAE: Flow-Friendly Self-Supervised Latents for Text-to-Motion Generation | [arXiv](https://arxiv.org/abs/2607.29180) | Uses Motion-JEPA-style pretraining and flow-friendly latent compression for text-to-motion generation. |
| T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data | [arXiv](https://arxiv.org/abs/2410.05016) | Applies JEPA to tabular feature subsets, predicting one feature-subset embedding from another without augmentations. |
| Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture | [arXiv](https://arxiv.org/abs/2603.20111) | Reinterprets JEPA through variational latent-variable modeling and applies it to tabular representation learning. |
| LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures | [arXiv](https://arxiv.org/abs/2509.14252) | Explores JEPA-style embedding-space objectives for language-model pretraining and finetuning. |
| VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language | [arXiv](https://arxiv.org/abs/2512.10942) / [ICLR page](https://mlanthology.org/iclr/2026/chen2026iclr-vljepa/) | Trains a vision-language model by predicting target text embeddings instead of autoregressive text tokens. |
| JEPA-T: Joint-Embedding Predictive Architecture with Text Fusion for Image Generation | [arXiv](https://arxiv.org/abs/2510.00974) | Adds text fusion and flow-matching alignment to a JEPA-style image-generation system. |
| Denoising with a Joint-Embedding Predictive Architecture | [arXiv](https://arxiv.org/abs/2410.03755) | D-JEPA adapts JEPA to autoregressive/generative image modeling with diffusion or flow losses over continuous tokens. |
| High-Resolution Image Synthesis via Next-Token Prediction | [arXiv](https://arxiv.org/abs/2411.14808) | D-JEPA-T2I extends D-JEPA toward high-resolution text-to-image synthesis using flow matching and continuous-resolution tricks. |
| Beyond Representation Learning: A Systematic Study of Joint-Embedding Predictive Generation for 3D Brain MRI | [arXiv](https://arxiv.org/abs/2608.28787) | Adapts D-JEPA-style predictive generation to 3D medical MRI synthesis and downstream medical tasks. |
| US-JEPA: A Joint Embedding Predictive Architecture for Medical Ultrasound | [arXiv](https://arxiv.org/abs/2602.19322) | Applies JEPA-style masked latent prediction to ultrasound representation learning with a static domain-specific teacher. |
| AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data | [arXiv](https://arxiv.org/abs/2501.04969) | Learns LiDAR BEV representations through JEPA-style latent prediction for autonomous-driving downstream tasks. |
| Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting | [arXiv](https://arxiv.org/abs/2602.12540) | AD-LiST-JEPA predicts future spatiotemporal LiDAR evolution for occupancy completion and forecasting. |
| JetParticle-JEPA: An Efficient Self-Supervised Representation Learning Method for Jet Tagging in High-Energy Physics | [arXiv](https://arxiv.org/abs/2606.14813) | Applies JEPA to continuous particle clouds for jet tagging, improving low-label and robustness behavior. |
| PI-JEPA: Label-Free Surrogate Pretraining for Coupled Multiphysics Simulation via Operator-Split Latent Prediction | [arXiv](https://arxiv.org/abs/2604.01349) | Uses JEPA-style masked latent prediction plus physics residuals to pretrain surrogate models with few expensive PDE solves. |
| Animal-JEPA: Advancing Animal Behavior Studies Through Joint Embedding Predictive Architecture in Video Analysis | [publisher record](https://collaborate.umb.edu/en/publications/animal-jepa-advancing-animal-behavior-studies-through-joint-embed/) | Adapts JEPA video representations to animal-behavior analysis. |
| PlayClass: Automated Play Behaviour Classification in Poultry | [arXiv](https://arxiv.org/abs/2605.27304) | Not a JEPA architecture paper, but directly uses V-JEPA 2.1 embeddings for animal-behavior classification. |
| FAR-JEPA: Factorized Autoregressive Process Laws for Resolving the Orthogonal Gauge in Joint Embeddings | [Preprints.org](https://www.preprints.org/manuscript/202607.2043) | Studies factorized autoregressive laws as a way to align JEPA latent coordinates with temporal process factors. |

## Architecture, optimization, data, and evaluation references cited around F-JEPA

| Paper / work | Link | Quick summary |
| --- | --- | --- |
| An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | [arXiv](https://arxiv.org/abs/2010.11929) | Vision Transformer foundation used by many JEPA visual encoders, including F-JEPA-style setups. |
| Attention Is All You Need | [NeurIPS](https://papers.nips.cc/paper/7181-attention-is-all-you-need) | Transformer architecture foundation for JEPA predictors and sequence dynamics models. |
| Scalable Diffusion Models with Transformers | [arXiv](https://arxiv.org/abs/2212.09748) | DiT/AdaLN conditioning reference relevant to F-JEPA's flow-time-conditioned Transformer predictor. |
| The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning | [Springer](https://link.springer.com/book/10.1007/978-1-4757-4321-0) | Optimization method used in latent action-sequence planning. |
| Model Predictive Heuristic Control: Applications to Industrial Processes | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/0005109878850018) | Classic model-predictive-control reference underlying F-JEPA-style receding-horizon planning. |
| DeepMind Control Suite | [arXiv](https://arxiv.org/abs/1801.00690) | Continuous-control benchmark suite; Reacher-like tasks appear in the F-JEPA evaluation neighborhood. |
| OGBench: Benchmarking Offline Goal-Conditioned RL | [arXiv](https://arxiv.org/abs/2410.20092) | Offline goal-conditioned RL benchmark used for OGBench-Cube-style evaluation. |
| A Test for Normality Based on the Empirical Characteristic Function | [DOI](https://doi.org/10.1093/biomet/70.3.723) | Statistical test behind the Gaussianity regularization used in SIGReg-style JEPA anti-collapse methods. |

## Non-paper resources found

| Resource | Link | Why it matters |
| --- | --- | --- |
| Flow-JEPA repository | [GitHub](https://github.com/HuoYanchen/Flow-JEPA) | Official code and experiment entry point for F-JEPA. |
| Pith review of Flow-JEPA | [Pith](https://pith.science/paper/2608.29029) | Useful external review that highlights the main claimed contribution and a key missing ablation: separating joint prediction from flow matching. |

