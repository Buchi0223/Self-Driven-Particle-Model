# 実装詳細ガイド

本ドキュメントでは MTM (Mixed Traffic Flow Model) の実装詳細を解説します。
論文の式番号と対応するコードの対応関係、設計判断の根拠、および既知の制約事項を記載します。

> **Reference**: Kanagaraj, V. & Treiber, M. (2018). *Self-Driven Particle Model for Mixed Traffic and Other Disordered Flows.* arXiv:1805.05076

---

## 目次

1. [アーキテクチャ概要](#1-アーキテクチャ概要)
2. [モジュール詳細](#2-モジュール詳細)
3. [論文の式とコードの対応表](#3-論文の式とコードの対応表)
4. [座標系と符号規約](#4-座標系と符号規約)
5. [パラメータ設計](#5-パラメータ設計)
6. [シミュレーションエンジン](#6-シミュレーションエンジン)
7. [検証シナリオ](#7-検証シナリオ)
8. [設計判断の記録](#8-設計判断の記録)
9. [既知の制約と今後の課題](#9-既知の制約と今後の課題)

---

## 1. アーキテクチャ概要

### データフロー

```
CFParams / MTMParams (parameters.py)
        │
        ▼
   Vehicle (vehicle.py)   ─── 初期状態設定
        │
        ▼
┌───────────────────────────────────────┐
│         Simulation Loop               │
│         (simulation.py)               │
│                                       │
│  for each vehicle:                    │
│    ├─ longitudinal_acceleration()     │
│    │    ├─ cf_free()         ← 式(6)  │
│    │    ├─ f_interaction()   ← 式(7)  │
│    │    │    ├─ alpha()      ← 式(8)  │
│    │    │    └─ cf_interaction()       │
│    │    └─ f_boundary()      ← 式(9)  │
│    │                                  │
│    └─ lateral_acceleration()          │
│         ├─ w0_desired()      ← 式(13) │
│         │    ├─ w0_ij_from_leader()   │
│         │    │    └─ alpha_tilde()    │
│         │    └─ w0_ij_from_follower() │
│         ├─ g_boundary()      ← 式(11) │
│         └─ (w0 - w) / τ     ← 式(12) │
│                                       │
│  euler_step() → 状態更新              │
└───────────────────────────────────────┘
        │
        ▼
  SimulationResult → 可視化 (scenarios/)
```

### 依存関係グラフ

```
parameters.py  ←── vehicle.py
     ↑               ↑
geometry.py    cf_models.py
     ↑    ↑         ↑
     │    └── longitudinal.py
     │              ↑
     └──── lateral.py
                ↑
         simulation.py ←── road.py
```

循環依存なし。下位モジュールは上位モジュールに依存しない。

---

## 2. モジュール詳細

### parameters.py — パラメータ定義

**責務**: 論文 Table 1 (車種別 CF) と Table 2 (MTM 固有) のパラメータを定義。

| クラス | 内容 | 特徴 |
|--------|------|------|
| `CFParams` | 車種別 CF パラメータ | frozen dataclass (不変) |
| `MTMParams` | MTM 固有 9 パラメータ | frozen dataclass, デフォルト値=Table 2 |

**車種別インスタンス**: `MOTORCYCLE`, `CAR`, `BUS`, `AUTO_RICKSHAW`

**設計判断**: `v0` は Table 1 の range から上限値を採用。分布サンプリングは将来課題。

### vehicle.py — 車両状態

**責務**: 動的状態 (x, y, v, w) + 物理属性 + CF パラメータを一体管理。

**mutable dataclass** を採用。理由: シミュレーション中に状態を in-place 更新するため。
`CFParams` (frozen) との対比で不変/可変の区別が明確。

**ファクトリメソッド**:
- `from_cf_params(cf_params, x, y, v, w)` — プログラム的生成
- `create(vehicle_type, x, y, v, w)` — 宣言的生成 (車種名文字列)

### geometry.py — 幾何計算

**責務**: Fig. 2 の幾何量を計算する純粋関数群。

| 関数 | 論文記号 | 説明 |
|------|---------|------|
| `delta_x(xi, xl)` | Δx_{il} | 縦方向距離 (front-center 間) |
| `delta_y(yi, yl)` | Δy_{il} | 横方向変位 (符号付き) |
| `W_bar(Wi, Wl)` | W̄_{il} | 平均車幅 |
| `sy_gap(dy, w_bar)` | s^y_{il} | 横方向ギャップ (式8) |
| `sx_gap(xi, xl, Ll)` | s_{il} | 縦方向ギャップ (バンパー間) |

全関数がステートレス・副作用なし。

### cf_models.py — Car-Following モデル

**責務**: IDM/ACC の加速度関数と free/interaction 分解 (式3-5)。

**IDM (Intelligent Driver Model)**:
```
a_CF = a · [1 - (v/v0)^δ - (s*/s)^2]
s*   = s0 + max(0, v·T + v·Δv / (2·√(a·b)))
```

**ACC (Adaptive Cruise Control)**:
IDM をベースに CAH (Constant Acceleration Heuristic) と coolness factor `c` でブレンド。
カットインへの過剰反応を抑制する。

**Free/Interaction 分解 (式3-5)**:
```python
cf_free(v)           = a · [1 - (v/v0)^δ]
cf_interaction(s,v,vl) = idm_acceleration(s,v,vl) - cf_free(v)
```

**エッジケース**: `gap ≤ 0 → -b_max` (緊急ブレーキ)

### road.py — 道路定義

**責務**: 道路境界の y 座標を保持。frozen dataclass。
デフォルト: `y_right=0, y_left=12` (チェンナイ観測の幅12m に対応)。

### longitudinal.py — MTM 縦方向ダイナミクス

**責務**: 式(2) の全縦方向加速度を計算。

**`alpha(dy, w_bar, sy0)` — 式(8)**:
横方向減衰因子。オーバーラップ (s^y < 0) → 1.0、クリアランス → exp(-s^y / s^y_0)。

**`find_most_interacting_leader()` — 式(2)**:
前方車両のみを候補とし、`argmax|f_il|` で最影響リーダーを選択。
理由: 複数リーダーの加算は過度に防御的な挙動を生む (論文 Section 2.1)。

**`f_boundary_longitudinal()` — 式(9)**:
`f = -b_b · (v/v0) · exp(-s^y / s^y_0b)`。
`v/v0` 因子により v=0 で力が消失 → 低速での狭路通過を許容。

### lateral.py — MTM 横方向ダイナミクス

**責務**: 式(10)-(16) の全横方向加速度を計算。最も複雑なモジュール。

**`alpha_tilde(dy, w_bar, sy0_tilde)` — 式(16)**:
符号付きの横方向減衰因子。`alpha()` との違い:
- **符号付き**: リーダーが左 → 正、右 → 負
- **オーバーラップ領域が線形**: `sign(Δy) · (1 + s^y / W̄)`
- Δy=0 で α̃=0 (インラインでは横方向力なし)

**`w0_ij_from_leader()` — 式(15)**:
```
w0_ij = λ · α̃(Δy) · a_CF_int · [1 - λ_Δw · (wj - wi) · sign(Δy)]
```
遅いリーダーが左にいる場合: α̃ > 0, a_CF_int < 0 → w0 < 0 (右に避ける)。

**`w0_ij_from_follower()` — actio=reactio (論文 p.8)**:
1. 役割を交換 (j がフォロワー, i がリーダー)
2. 符号を反転
3. politeness factor `p` で重み付け

**`lateral_acceleration()` — 式(10), (12)**:
```
g_i = g_boundary + (w0_desired - w_i) / τ
```
`g_self = 0` (戦術的成分は本論文では未考慮)。

### simulation.py — シミュレーションエンジン

**責務**: 全モジュールを結合し、Euler 法で時間積分。

**`euler_step()` の制約適用**:
1. `v ≥ 0` (後退禁止)
2. `|w/v| ≤ tan(θ)` (heading 制約)
3. `v = 0 → w = 0` (停止時は横方向速度もゼロ)

**位置更新**: semi-implicit Euler (`x += v_new · dt`) で安定性を向上。

---

## 3. 論文の式とコードの対応表

| 式番号 | 内容 | ファイル | 関数 |
|--------|------|---------|------|
| (2) | 全縦方向加速度 | longitudinal.py | `longitudinal_acceleration()` |
| (3)-(5) | Free/Interaction 分解 | cf_models.py | `cf_free()`, `cf_interaction()` |
| (6) | 自己駆動力 | cf_models.py | `cf_free()` |
| (7) | 縦方向相互作用 | longitudinal.py | `f_interaction_single()` |
| (8) | 横方向減衰因子 α | longitudinal.py | `alpha()` |
| (9) | 道路境界の縦方向力 | longitudinal.py | `f_boundary_longitudinal()` |
| (10) | 全横方向加速度 | lateral.py | `lateral_acceleration()` |
| (11) | 道路境界の横方向力 | lateral.py | `g_boundary_lateral()` |
| (12) | OV-like 横方向緩和 | lateral.py | `lateral_acceleration()` 内 |
| (13) | Desired lateral speed 合計 | lateral.py | `w0_desired()` |
| (15) | ペア横方向 desired speed | lateral.py | `w0_ij_from_leader()` |
| (16) | 横方向減衰因子 α̃ | lateral.py | `alpha_tilde()` |

---

## 4. 座標系と符号規約

```
        y_left (= 12.0)
    ────────────────────── 左境界
    │                    │
    │   y 増加方向 ↑     │
    │                    │
    │   → x (進行方向)   │
    │                    │
    ────────────────────── 右境界
        y_right (= 0.0)
```

- **x**: 縦方向 (道路進行方向、正が前方)
- **y**: 横方向 (正が左方向) — **注意: 論文は "y increasing to the right"**
- 位置は車両の **front-center** を基準
- 車両は **矩形** (Fig. 2)

**境界力の符号**:
- 右境界接近 → 左に押す (g > 0, y 増加方向)
- 左境界接近 → 右に押す (g < 0, y 減少方向)

> **Issue #19**: 座標系の論文準拠検証が未完。現状は内部的に一貫しているが、論文との厳密な対応は要検証。

---

## 5. パラメータ設計

### CF パラメータ (Table 1)

| 車種 | length | width | v0 | T | s0 | a | b |
|------|--------|-------|----|---|----|---|---|
| Motorcycle | 1.8 | 0.6 | 25 | 0.3 | 0.5 | 2.0 | 2.0 |
| Car | 4.2 | 1.7 | 18 | 0.8 | 2.0 | 1.0 | 1.0 |
| Bus | 10.3 | 2.1 | 14 | 1.0 | 2.0 | 1.0 | 1.0 |
| Auto-Rickshaw | 2.6 | 0.9 | 6 | 1.0 | 2.0 | 1.0 | 1.0 |

全車種共通: `b_max = 9.0 m/s²`, `delta = 4`, `coolness = 0.99`

### MTM パラメータ (Table 2)

| パラメータ | 記号 | 値 | Python変数名 |
|-----------|------|-----|-------------|
| 最大道路軸角度 | θ | 0.2 rad | `theta` |
| 横方向減衰(制動) | s^y_0 | 0.15 m | `sy0` |
| 横方向減衰(操舵) | s̃^y_0 | 0.30 m | `sy0_tilde` |
| 境界減衰(制動) | s^y_{0b} | 0.15 m | `sy0b` |
| 境界減衰(操舵) | s̃^y_{0b} | 0.25 m | `sy0b_tilde` |
| 横方向感度 | λ | 0.4 s | `lam` |
| 横方向時定数 | τ | 1.0 s | `tau` |
| 横方向相対速度感度 | λ_{Δw} | 0.7 s/m | `lam_dw` |
| Politeness factor | p | 0.2 | `p` |

**論文未明記のパラメータ** (推定値):
- `bb = 3.0 m/s²` — 境界制動減速度
- `bb_tilde = 5.0 m/s²` — 境界横方向加速度 (操舵 > 制動)
- `a_thr = 0.01 m/s²` — 相互作用閾値 (プレ検証用)

---

## 6. シミュレーションエンジン

### 時間積分

**手法**: Semi-implicit Euler 法
```
v_new = v + f · dt        (速度更新)
v_new = max(v_new, 0)     (後退禁止)
w_new = clamp(w_new)      (heading 制約)
x += v_new · dt           (位置更新: 更新後の速度を使用)
y += w_new · dt
```

**タイムステップ**: `dt = 0.05 s` (交通マイクロシミュレーションの標準的な値)

### 計算量

- 加速度計算: O(N²) — 全車両ペアの相互作用
- プレ検証 (2-3台): 問題なし
- 大規模 (数十台以上): 空間インデックスが必要

---

## 7. 検証シナリオ

### Step 9: 加速度ベクトル場 (Fig.3)

(x, y) グリッド上で静的に加速度を計算。2台のリーダーに対する follower の応答を可視化。
- (a) リーダー間通過可能: 矢印がギャップを通る方向
- (b) リーダー近接で迂回: 矢印がリーダー群を避ける方向

### Step 10: 単独追い越し

最も基本的な動的テスト。遅い1台のリーダーを追い越す。
確認: 減速→横方向回避→加速→直進復帰

### Step 11: 2台の間を通過 (Fig.5a)

L1(y=3), L2(y=9) の間を F が通過。十分な横方向ギャップがある場合。

### Step 12: 2台を迂回 (Fig.5b)

L1(y=5.5), L2(y=7.5) が近接。F(y=3) が右側から迂回して追い越し。

---

## 8. 設計判断の記録

### frozen vs mutable dataclass

| クラス | 種類 | 理由 |
|--------|------|------|
| CFParams | frozen | テンプレートとして不変であるべき |
| MTMParams | frozen | シミュレーション中にパラメータ変更しない |
| Road | frozen | 道路形状は固定 |
| Vehicle | mutable | 動的状態 (x,y,v,w) がシミュレーション中に更新される |

### CF パラメータをフラット展開

Vehicle 内で `veh.v0` と直接アクセス可能にした。`veh.cf_params.v0` とたどる必要がなく、シミュレーションループの可読性・パフォーマンスに寄与。

### cf_models.py を Vehicle に依存させない

Car-Following モデルは純粋な数学関数として実装。Vehicle や CFParams への依存なし。
結合は上位モジュール (longitudinal.py, lateral.py) の責務。

### リーダー選択で前方車両のみ候補

`find_most_interacting_leader()` は `xl > xi` の車両のみを候補とする。
理由: 後方車両を含めると縦方向で過度に防御的な挙動になる (論文 Section 2.1)。

### g_self = 0

論文の戦術的成分 (レーン変更意図、入退出) は本実装では未考慮。
"will not be considered in this paper" (論文 p.6)。

---

## 9. 既知の制約と今後の課題

### 座標系 (Issue #19)
本実装は y が左方向に増加。論文は右方向。境界力符号で調整済みだが厳密検証が必要。

### CAH 分岐整理 (Issue #20)
`_cah_acceleration()` に冗長な条件分岐。動作に影響なし。

### 浮動小数点タイミング (Issue #21)
`run_simulation()` の記録判定が浮動小数点比較。長時間シミュレーションでリスクあり。

### テスト追加 (Issue #22)
混合車種シナリオ、長時間安定性、follower 統合テストが不足。

### 将来の拡張
- ACC (coolness factor) の効果検証
- v0 の分布サンプリング (Table 1 の range)
- Floor Field によるレーンマーカーの導入
- 論文 Section 4 の Chennai シミュレーション再現
