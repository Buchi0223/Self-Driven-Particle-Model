# Self-Driven Particle Model

## Project Overview
Self-Driven Particle Model for Mixed Traffic and Other Disordered Flows の実装プロジェクト。
論文: Kanagaraj & Treiber (2018), arXiv:1805.05076 (`paper/1805.05076.pdf`)

混合交通流（レーン規律のない途上国の交通など）をシミュレーションする2次元微視的交通モデル（MTM: Mixed Traffic Flow Model）を実装する。

## Key Concepts
- **MTM (Mixed Traffic Flow Model)**: 従来の1次元 car-following モデルを2次元力場に一般化
- **縦方向ダイナミクス**: car-following モデル（IDM/ACC/OVM/Gipps）ベースの加速度 + 横方向減衰
- **横方向ダイナミクス**: MOBIL車線変更モデルを連続座標に一般化した OV-like モデル
- **Social Force**: 自己駆動力 + 車両間相互作用力 + 境界力の重ね合わせ

## Model Equations (Quick Reference)
- 全加速度: `dv_i/dt = f_self + f_int(l') + Σ f_ib` (式1,2)
- 縦方向相互作用: `f_il' = α(Δy) * a_CF_int(Δx, v_i, v_l)` (式7)
- 横方向減衰: `α(Δy) = min(exp(-s^y / s^y_0), 1)` (式8)
- 横方向ダイナミクス: `g_int = (w0_i - w_i) / τ` (式12)
- 横方向desired速度: `w0_ij = λ * α̃(Δy) * a_CF_int * [1 - λ_Δw(w_j - w_i)sign(Δy)]` (式15)

## Model Parameters (Table 2 in paper)
MTM固有パラメータ（9個）: s^y_0, s^y_0b, s̃^y_0, s̃^y_0b, b_b, b̃_b, λ, λ_Δw, τ
+ underlying CF モデルのパラメータ（IDM/ACC: v0, T, s0, a, b, δ, coolness）

## Tech Stack
- Python 3
- シミュレーション可視化: matplotlib

## Commands
```bash
# (TBD: プロジェクト進行に応じて追記)
```

## Directory Structure
```
paper/           # 参考論文 PDF
```

## Development Guidelines
- 論文の式番号をコード中のコメントで参照すること
- SI単位系を使用（m, s, m/s, m/s²）
- 車両は矩形オブジェクトとして扱う（Fig. 2参照）
