# Mixed-Precision Quantization evaluation
  - HAWQ2_fisher (greedy via empirical Fisher)(An unofficial implementation of [HAWQ-V2](https://arxiv.org/pdf/1911.03852) using [PyHessian](https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py)) 
  - OT_HAWQ_like (Sinkhorn OT with size-weighted marginals)
  - OT_Fisher_Critical (same API; if no critical classes -> same as OT_HAWQ_like)
  - DiffSinkhornDynamic (differentiable Sinkhorn with STE mixture + short QAT)
  - SinkhornMCKPDynamic (MCKP (Chen et al. quadratic cost) + short QAT)
  - ILP(integer linear programming ) 
# Usage
> python eval_quant_methods.py --train_cifar10 --model resnet18 --train_epochs 2 --methods HAWQ2_fisher OT_HAWQ_like DiffSinkhornDynamic SinkhornMCKPDynamic --out_dir runs/cifar_res18

> python eval_quant_methods.py --model resnet18 --imagenet_val /path/to/imagenet/val --methods HAWQ2_fisher OT_HAWQ_like --out_dir runs/imagenet_res18

> python eval_quant_methods.py --train_cifar10 --bits 2 4 8 --avg_bits 4.5 --dynamic_steps 80

# comment
- 重みのみ（per-tensor）量子化。精度を突き詰めるには activation 量子化や ACIQ/LSQ によるクリッピング推定、per-channel 量子化の追加を推奨。
- OT_Fisher_Critical は DETR などでクリティカル目的を組むと効果が出ます（分類で未指定なら 2) と同等）。
- DiffSinkhornDynamic/SinkhornMCKPDynamic は短時間 QAT（--dynamic_steps）で割当を学習→argmax で固定して評価しています。長めに回すと安定する
- 速度重視なら --methods で方式を絞る
- 必要に応じてactivation 量子化+ACIQ 版や DETR用loss callback（ハンガリアン + クリティカル目的）を追加

# 用語
- ILP integer linear programming 
- MCKP Multiple-Choice Knapsack
- ACIQ (Analytical Clipping for Integer Quantization)
- LSQ (Learned Step Size Quantization)
- [ハンガリアン（Kuhn–Munkres）](https://en.wikipedia.org/wiki/Hungarian_algorithm)
DETR系で使う 1対1対応 の割り当て（バイパーティトマッチング）手法。各予測クエリと各GT（＋不要分の“空集合”）のコスト行列を作り、総コスト最小になる対応を見つける。
- クリティカル
重要クラスだけ特に落としたくない”という方針を損失に反映させる手法

# Reference
- [ILP](https://github.com/1hunters/LIMPQ)
- [HAWQ-V2](https://arxiv.org/pdf/1911.03852) 
- [PyHessian](https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py)
- [HAWQ-V3](https://arxiv.org/abs/2011.10680)
- [Optimal transport](https://arxiv.org/abs/2412.15195)
- [MCKP](https://arxiv.org/abs/2110.06554)
- [Fisher-aware Quantization for DETR Detectors with Critical-category Objectives](https://arxiv.org/abs/2407.03442)
