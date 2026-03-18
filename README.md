# Self-Driven Particle Model for Mixed Traffic

混合交通流（レーン規律のない途上国の交通など）をシミュレーションする 2 次元微視的交通モデル **MTM (Mixed Traffic Flow Model)** の Python 実装です。

> **Reference**: Kanagaraj, V. & Treiber, M. (2018). *Self-Driven Particle Model for Mixed Traffic and Other Disordered Flows.* arXiv:1805.05076

## What is MTM?

従来の 1 次元 Car-Following モデル（IDM/ACC）を **2 次元の力場** に一般化したモデルです。歩行者の Social Force Model と類似した枠組みですが、高速走行時の運動学的制約（制動距離、衝突回避）を明示的に考慮しています。

**主な特徴:**
- インライン追従時は従来の Car-Following モデルに帰着
- 低速時は歩行者 Social Force モデルに帰着
- レーンマーカーの Floor Field を追加すると車線変更モデルに帰着

```
全加速度 = 自己駆動力 + 車両間相互作用力 + 道路境界力
         = f_self    + f_interaction     + f_boundary
```

## Quick Start

```bash
# 依存関係インストール
uv sync --extra dev

# 全検証シナリオを実行 → output/ に画像が生成されます
uv run python run_scenario.py all

# テスト実行 (150件)
uv run pytest
```

### 個別シナリオ実行

```bash
uv run python run_scenario.py accel_field     # 加速度ベクトル場 (論文 Fig.3)
uv run python run_scenario.py single_pass     # 単独追い越し
uv run python run_scenario.py pass_between    # 2台の間を通過 (論文 Fig.5a)
uv run python run_scenario.py circumvent      # 2台を迂回 (論文 Fig.5b)
```

## Project Structure

```
src/
  parameters.py      # パラメータ定義 (論文 Table 1, 2)
  vehicle.py         # 車両状態クラス
  geometry.py        # 幾何計算ヘルパー (Δx, Δy, gap)
  cf_models.py       # Car-Following モデル (IDM, ACC)
  road.py            # 道路境界定義
  longitudinal.py    # MTM 縦方向ダイナミクス (式 2, 6-9)
  lateral.py         # MTM 横方向ダイナミクス (式 10-16)
  simulation.py      # 時間積分ループ (Euler法)
tests/               # ユニットテスト (150件)
scenarios/            # 検証シナリオ・可視化
paper/                # 参考論文 PDF
output/               # シナリオ出力画像
```

## Verification Results

| シナリオ | 検証内容 | 結果 |
|---------|---------|------|
| 加速度ベクトル場 | 論文 Fig.3 と定性的比較 | リーダー後方の減速域・迂回方向が一致 |
| 単独追い越し | 減速→回避→加速の流れ | 正常な追い越し挙動 |
| 2台の間を通過 | 論文 Fig.5a に対応 | ギャップを通過して追い越し成功 |
| 2台を迂回 | 論文 Fig.5b に対応 | 近接リーダー群を迂回して追い越し成功 |

## Tech Stack

- **Python 3.12** / **uv** (パッケージ管理)
- **numpy** / **matplotlib** (数値計算・可視化)
- **pytest** (テスト)

## License

This is a research implementation for academic purposes.
