#!/usr/bin/env python
"""
Brute Force Ensemble - 暴力穷举组合回测 + 结果分析

将所有模型的预测结果进行暴力组合，对每个组合做等权融合+回测，
最后对结果进行全面分析（模型归因、相关性、聚类、优化权重等）。

运行方式：
  cd QuantPits && python quantpits/scripts/brute_force_ensemble.py

常用命令：
  # 快速测试（最多3个模型的组合）
  python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

  # 仅分析已有结果（不重新跑回测）
  python quantpits/scripts/brute_force_ensemble.py --analysis-only

  # 从上次中断处继续
  python quantpits/scripts/brute_force_ensemble.py --resume

  # 只跑回测、跳过分析
  python quantpits/scripts/brute_force_ensemble.py --skip-analysis

  # 使用模型分组穷举 (每组只选一个) — 大幅减少组合数
  python quantpits/scripts/brute_force_ensemble.py --use-groups

  # 指定自定义分组配置
  python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml
"""

import os
import sys
import json
import gc
import signal
import itertools
import logging
import argparse
import yaml
import warnings
from datetime import datetime
from collections import Counter
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

from quantpits.utils import env
os.chdir(env.ROOT_DIR)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
# 已在上方导入并切换目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# BACKTEST_CONFIG has been migrated to strategy provider `config/strategy_config.yaml`

# ---------------------------------------------------------------------------
# 全局中断标志 & 信号处理
# ---------------------------------------------------------------------------
_shutdown = False
_original_sigint = None
_original_sigterm = None


def _signal_handler(signum, frame):
    """收到 SIGINT/SIGTERM 后标记安全中断"""
    global _shutdown
    if _shutdown:
        # 第二次中断 → 强制退出
        print("\n\n⛔ 再次收到中断信号，强制退出...")
        sys.exit(1)
    _shutdown = True
    sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n\n⚠️  收到 {sig_name}，将在当前批次完成后安全退出...")
    print("   (再次按 Ctrl+C 强制退出)")


def _install_signal_handlers():
    """安装信号处理器"""
    global _original_sigint, _original_sigterm
    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _restore_signal_handlers():
    """恢复原始信号处理器"""
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)
    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)


# ============================================================================
# Stage 0: 初始化 & 配置加载
# ============================================================================

def init_qlib():
    """初始化 Qlib（委托给 env.init_qlib）"""
    env.init_qlib()


def load_config(record_file="latest_train_records.json"):
    """使用 config_loader 加载统一配置"""
    from quantpits.utils.config_loader import load_workspace_config
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            train_records = json.load(f)
    else:
        train_records = {"models": {}, "experiment_name": "unknown"}

    model_config = load_workspace_config(ROOT_DIR)

    return train_records, model_config


# ============================================================================
# Stage 1: 加载预测数据
# ============================================================================

def load_predictions(train_records):
    """
    从 Qlib Recorder 加载所有模型的预测值，归一化后返回宽表。
    """
    from quantpits.utils.predict_utils import load_predictions_from_recorder
    norm_df, model_metrics, _ = load_predictions_from_recorder(train_records)
    return norm_df, model_metrics


# ============================================================================
# Stage 2: 相关性分析
# ============================================================================

def correlation_analysis(norm_df, output_dir, anchor_date):
    """计算并保存预测值相关性矩阵"""
    print(f"\n{'='*60}")
    print("Stage 2: 相关性分析")
    print(f"{'='*60}")

    corr_matrix = norm_df.corr()
    print("\n=== 模型预测相关性矩阵 ===")
    print(corr_matrix.round(4))

    # 保存
    corr_path = os.path.join(output_dir, f"correlation_matrix_{anchor_date}.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\n相关性矩阵已保存: {corr_path}")

    return corr_matrix


def split_is_oos_by_args(norm_df, args):
    """根据参数将 norm_df 划分为 IS (In-Sample) 和 OOS (Out-of-Sample)"""
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    
    dates = norm_df.index.get_level_values("datetime").unique().sort_values()
    max_date = dates.max()
    
    cutoff_date = max_date
    if args.exclude_last_years > 0:
        cutoff_date = cutoff_date - pd.DateOffset(years=args.exclude_last_years)
    if args.exclude_last_months > 0:
        cutoff_date = cutoff_date - pd.DateOffset(months=args.exclude_last_months)
        
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    if end_date and end_date < cutoff_date:
        cutoff_date = end_date
        
    is_mask = norm_df.index.get_level_values("datetime") <= cutoff_date
    if start_date:
        is_mask &= norm_df.index.get_level_values("datetime") >= start_date
        
    is_norm_df = norm_df[is_mask]
    
    # OOS 是截止日之后的数据（如果有指定 start_date/end_date，暂不限制 OOS 的末尾跨度，或只保留剩下的）
    oos_mask = norm_df.index.get_level_values("datetime") > cutoff_date
    oos_norm_df = norm_df[oos_mask]
    
    print(f"\n=== 数据集划分 (In-Sample / Out-Of-Sample) ===")
    if not is_norm_df.empty:
        print(f"IS 期  : {is_norm_df.index.get_level_values('datetime').min().date()} ~ {is_norm_df.index.get_level_values('datetime').max().date()} (共 {len(is_norm_df.index.get_level_values('datetime').unique())} 天)")
    else:
        print("IS 期  : 无数据")
        
    if not oos_norm_df.empty:
        print(f"OOS 期 : {oos_norm_df.index.get_level_values('datetime').min().date()} ~ {oos_norm_df.index.get_level_values('datetime').max().date()} (共 {len(oos_norm_df.index.get_level_values('datetime').unique())} 天)")
    else:
        print("OOS 期 : 无数据")
        
    return is_norm_df, oos_norm_df


# ============================================================================
# Stage 2.5: 模型分组 & 组合生成
# ============================================================================

def load_combo_groups(group_config_path, available_models):
    """
    加载分组配置，验证模型名，返回有效分组。

    Args:
        group_config_path: combo_groups.yaml 路径
        available_models: 当前加载到的模型列表 (norm_df.columns)

    Returns:
        groups: dict, group_name -> list of valid model names
    """
    with open(group_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_groups = cfg.get("groups", {})
    if not raw_groups:
        raise ValueError(f"分组配置为空: {group_config_path}")

    available_set = set(available_models)
    groups = {}
    skipped_models = []

    for gname, models in raw_groups.items():
        valid = [m for m in models if m in available_set]
        invalid = [m for m in models if m not in available_set]
        if invalid:
            skipped_models.extend(invalid)
            print(f"  ⚠️  组 [{gname}] 中以下模型不存在于预测数据中，已忽略: {invalid}")
        if valid:
            groups[gname] = valid
        else:
            print(f"  ⚠️  组 [{gname}] 无有效模型，已跳过")

    if skipped_models:
        print(f"  共忽略 {len(skipped_models)} 个无效模型")

    # 检查未分组的模型 (仅打印提示，不自动参与)
    grouped_models = set()
    for models in groups.values():
        grouped_models.update(models)
    ungrouped = available_set - grouped_models
    if ungrouped:
        print(f"  ℹ️  以下模型未在任何分组中，将被排除: {sorted(ungrouped)}")

    return groups


def generate_grouped_combinations(groups, min_combo_size=1, max_combo_size=0):
    """
    基于分组生成组合：从所有组的子集中，每组选一个模型，做笛卡尔积。

    为支持 min/max combo size，我们枚举组的子集（选哪些组参与），
    然后对参与的组做 itertools.product。

    Args:
        groups: dict, group_name -> list of models
        min_combo_size: 最小组合大小 (选几个组)
        max_combo_size: 最大组合大小 (0=全部组)

    Returns:
        list of tuples, 每个 tuple 是一个模型组合
    """
    group_names = list(groups.keys())
    n_groups = len(group_names)
    max_size = max_combo_size if max_combo_size > 0 else n_groups
    max_size = min(max_size, n_groups)

    all_combinations = []

    # 枚举选哪些组参与 (选 r 个组的组合)
    for r in range(min_combo_size, max_size + 1):
        for group_subset in itertools.combinations(group_names, r):
            # 对选中的组做笛卡尔积
            model_lists = [groups[g] for g in group_subset]
            for combo in itertools.product(*model_lists):
                all_combinations.append(combo)

    return all_combinations


# ============================================================================
# Stage 3: 暴力穷举回测
# ============================================================================

def run_single_backtest(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end, st_config=None, bt_config=None
):
    """对指定的模型组合进行回测，返回指标字典或 None"""
    from quantpits.utils import strategy
    from quantpits.utils.backtest_utils import run_backtest_with_strategy, standard_evaluate_portfolio

    if st_config is None:
        st_config = strategy.load_strategy_config()
    if bt_config is None:
        bt_config = strategy.get_backtest_config(st_config)

    # 1. 合成信号 (等权均值，归一化后的) (仅在当前组合子集上求交集 dropna)
    combo_score = norm_df[list(combo_models)].dropna(how='any').mean(axis=1)

    import copy
    st_config = copy.deepcopy(st_config)
    st_config["strategy"]["params"]["topk"] = top_k
    st_config["strategy"]["params"]["n_drop"] = drop_n

    strategy_inst = strategy.create_backtest_strategy(combo_score, st_config)

    # 2. 回测
    try:
        report, _ = run_backtest_with_strategy(
            strategy_inst=strategy_inst,
            trade_exchange=trade_exchange,
            freq=freq,
            account_cash=bt_config["account"],
            bt_start=bt_start,
            bt_end=bt_end
        )

        # 3. 标准化计算结果
        st_config_inner = strategy.load_strategy_config()
        benchmark_col = st_config_inner.get('benchmark', 'SH000300')
        
        metrics = standard_evaluate_portfolio(report, benchmark_col, freq)

        return {
            "models": ",".join(combo_models),
            "n_models": len(combo_models),
            "Ann_Ret": metrics.get("CAGR", 0),
            "Max_DD": metrics.get("Max_Drawdown", 0),
            "Excess_Ret": metrics.get("Absolute_Return", 0) - metrics.get("Benchmark_Absolute_Return", 0),
            "Ann_Excess": metrics.get("Excess_Return_CAGR", 0),
            "Total_Ret": metrics.get("Absolute_Return", 0),
            "Final_NAV": report.iloc[-1]["account"],
            "Calmar": metrics.get("Calmar", 0) if not pd.isna(metrics.get("Calmar")) else 0,
        }
    except Exception as e:
        print(f"  [ERROR] Combo {combo_models} failed: {e}")
        # import traceback
        # traceback.print_exc()
        return None


def _append_results_to_csv(csv_path, results, write_header=False):
    """将一批结果追加写入 CSV 文件"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def brute_force_backtest(
    norm_df, top_k, drop_n, benchmark, freq,
    min_combo_size, max_combo_size, output_dir, anchor_date, resume=False,
    n_jobs=4, use_groups=False, group_config=None,
    batch_size=50,
):
    """
    暴力穷举所有模型组合并回测。

    支持:
    - 分批执行 + 增量保存 (防止崩溃丢失进度)
    - SIGINT/SIGTERM 安全中断
    - 模型分组穷举 (--use-groups)

    Returns:
        results_df: DataFrame，所有组合的回测结果
    """
    global _shutdown
    _shutdown = False

    from qlib.backtest.exchange import Exchange

    print(f"\n{'='*60}")
    print("Stage 3: 暴力穷举回测 (Batched Threading + Checkpoint)")
    print(f"{'='*60}")

    model_candidates = list(norm_df.columns)

    # 准备共享的 Exchange 对象
    print("Initializing Shared Exchange...")
    bt_start = str(norm_df.index.get_level_values(0).min().date())
    bt_end = str(norm_df.index.get_level_values(0).max().date())
    all_codes = sorted(norm_df.index.get_level_values(1).unique().tolist())

    from quantpits.utils import strategy
    st_config = strategy.load_strategy_config()
    bt_config = strategy.get_backtest_config(st_config)
    
    exchange_kwargs = bt_config["exchange_kwargs"].copy()
    exchange_freq = exchange_kwargs.pop("freq", "day")

    trade_exchange = Exchange(
        freq=exchange_freq,
        start_time=bt_start,
        end_time=bt_end,
        codes=all_codes,
        **exchange_kwargs
    )
    print(f"Shared Exchange Initialized. Period: {bt_start} ~ {bt_end}, Instruments: {len(all_codes)}")

    # ── 生成组合 ──
    if use_groups and group_config:
        print(f"\n📦 分组穷举模式 (配置: {group_config})")
        groups = load_combo_groups(group_config, model_candidates)
        print(f"有效分组 ({len(groups)}个):")
        total_product = 1
        for gname, models in groups.items():
            print(f"  [{gname}] ({len(models)}个): {models}")
            total_product *= len(models)

        all_combinations = generate_grouped_combinations(
            groups, min_combo_size, max_combo_size
        )
        print(f"\n分组笛卡尔积组合数: {len(all_combinations)}")
    else:
        max_size = max_combo_size if max_combo_size > 0 else len(model_candidates)
        max_size = min(max_size, len(model_candidates))
        print(f"待穷举模型 ({len(model_candidates)}个): {model_candidates}")
        print(f"组合大小范围: {min_combo_size} ~ {max_size}")

        all_combinations = []
        for r in range(min_combo_size, max_size + 1):
            all_combinations.extend(itertools.combinations(model_candidates, r))

    print(f"总组合数: {len(all_combinations)}")
    print(f"回测频率: {freq}, TopK={top_k}, DropN={drop_n}")
    print(f"并发线程数: {n_jobs}, 批次大小: {batch_size}")

    # ── Resume: 加载已有结果 ──
    csv_path = os.path.join(output_dir, f"brute_force_results_{anchor_date}.csv")
    done_combos = set()

    if resume and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_combos = set(existing_df["models"].tolist())
        print(f"Resume 模式: 已有 {len(done_combos)} 个组合，跳过")

    # 过滤已完成的组合
    pending = [
        c for c in all_combinations if ",".join(c) not in done_combos
    ]
    print(f"待回测组合数: {len(pending)}")

    if not pending:
        print("所有组合已完成！")
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame()
    else:
        # 如果不是 resume，先写一个空的带 header 的 CSV
        need_header = not (resume and os.path.exists(csv_path))
        if need_header:
            # 写 header
            header_cols = [
                "models", "n_models", "Ann_Ret", "Max_DD",
                "Excess_Ret", "Ann_Excess", "Total_Ret", "Final_NAV", "Calmar"
            ]
            pd.DataFrame(columns=header_cols).to_csv(csv_path, index=False)

        # 临时静默 Qlib 日志
        qlib_log = logging.getLogger("qlib")
        original_level = qlib_log.level
        qlib_log.setLevel(logging.WARNING)

        # 安装信号处理器
        _install_signal_handlers()

        completed_count = len(done_combos)
        total_count = len(all_combinations)
        failed_count = 0

        # ── 分批处理 ──
        pbar = tqdm(
            total=len(pending),
            desc=f"Brute Force (Threads={n_jobs})",
            unit="combo",
        )

        try:
            for batch_start in range(0, len(pending), batch_size):
                if _shutdown:
                    break

                batch = pending[batch_start : batch_start + batch_size]
                batch_results = []

                # 使用 ThreadPoolExecutor 处理当前批次
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_combo = {
                        executor.submit(
                            run_single_backtest,
                            combo,
                            norm_df,
                            top_k,
                            drop_n,
                            benchmark,
                            freq,
                            trade_exchange,
                            bt_start,
                            bt_end,
                            st_config,
                            bt_config
                        ): combo
                        for combo in batch
                    }

                    for future in as_completed(future_to_combo):
                        if _shutdown:
                            # 收到中断，不再等待其他 future
                            # 但已提交的会继续完成 (ThreadPoolExecutor 的行为)
                            pass
                        try:
                            res = future.result()
                            if res:
                                batch_results.append(res)
                                completed_count += 1
                            else:
                                failed_count += 1
                        except Exception:
                            failed_count += 1
                        pbar.update(1)

                # 批次完成 → 增量写入 CSV
                if batch_results:
                    _append_results_to_csv(csv_path, batch_results, write_header=False)

                # 释放内存
                del batch_results
                gc.collect()

                if _shutdown:
                    break

        finally:
            pbar.close()
            _restore_signal_handlers()
            qlib_log.setLevel(original_level)

        # 打印完成/中断状态
        if _shutdown:
            print(f"\n⚠️  已安全中断！")
            print(f"   已完成: {completed_count}/{total_count} 组合")
            print(f"   失败: {failed_count} 个")
            print(f"   结果已保存至: {csv_path}")
            print(f"   使用 --resume 继续未完成的组合")
        else:
            print(f"\n✅ 回测全部完成！")
            print(f"   有效: {completed_count - len(done_combos)}, 失败: {failed_count}")

        # 读取完整结果 (包括 resume 的)
        results_df = pd.read_csv(csv_path)

    # 排序并重新保存
    if not results_df.empty:
        results_df = results_df.sort_values(
            by="Ann_Excess", ascending=False
        ).reset_index(drop=True)
        results_df.to_csv(csv_path, index=False)
        print(f"结果已排序保存: {csv_path} ({len(results_df)} 条)")
    else:
        print("警告: 无有效回测结果")

    return results_df


# ============================================================================
# Main
# ============================================================================

def main():
    from quantpits.utils import env
    env.safeguard("Brute Force Ensemble")
    parser = argparse.ArgumentParser(
        description="暴力穷举模型组合回测 + 结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试 (最多 3 个模型)
  python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

  # 从中断处继续
  python quantpits/scripts/brute_force_ensemble.py --resume
        """,
    )
    parser.add_argument(
        "--record-file", type=str, default="latest_train_records.json",
        help="训练记录文件路径 (默认: latest_train_records.json)",
    )
    parser.add_argument(
        "--training-mode", type=str, default=None,
        choices=["static", "rolling"],
        help="训练模式过滤 (默认 None=全部，static 或 rolling)",
    )
    parser.add_argument(
        "--max-combo-size", type=int, default=0,
        help="最大组合大小 (0=全部, 默认: 0)",
    )
    parser.add_argument(
        "--min-combo-size", type=int, default=1,
        help="最小组合大小 (默认: 1)",
    )
    parser.add_argument(
        "--freq", type=str, default=None, choices=["day", "week"],
        help="回测交易频率 (默认: 从 model_config 读取)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/brute_force",
        help="输出目录 (默认: output/brute_force)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="回测开始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="回测结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--exclude-last-years", type=int, default=0,
        help="在 IS 阶段排除最后 N 年的数据（留作 OOS）",
    )
    parser.add_argument(
        "--exclude-last-months", type=int, default=0,
        help="在 IS 阶段排除最后 N 个月的数据（留作 OOS）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从已有 CSV 继续 (跳过已完成的组合)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4,
        help="并发回测线程数 (默认: 4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="每批处理的组合数 (默认: 50，影响 checkpoint 粒度和内存占用)",
    )
    parser.add_argument(
        "--use-groups", action="store_true",
        help="启用分组穷举模式 (每组只选一个模型)",
    )
    parser.add_argument(
        "--group-config", type=str, default="config/combo_groups.yaml",
        help="分组配置文件路径 (默认: config/combo_groups.yaml)",
    )
    args = parser.parse_args()

    # 初始化
    print("=" * 60)
    print("Brute Force Ensemble - 暴力穷举组合回测")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    init_qlib()
    
    train_records, model_config = load_config(args.record_file)
    
    # 确定频率
    freq = args.freq or model_config.get("freq", "week")
    args.freq = freq
    print(f"当前交易频率: {freq}")
    
    anchor_date = train_records.get(
        "anchor_date", datetime.now().strftime("%Y-%m-%d")
    )
    top_k = model_config.get("TopK", 22)
    drop_n = model_config.get("DropN", 3)
    benchmark = model_config.get("benchmark", "SH000300")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: 加载预测数据 (应用 training-mode 过滤)
    if getattr(args, 'training_mode', None):
        from quantpits.utils.train_utils import filter_models_by_mode
        filtered = filter_models_by_mode(train_records.get('models', {}), args.training_mode)
        train_records = dict(train_records)
        train_records['models'] = filtered
        print(f"训练模式过滤: {args.training_mode} (剩余 {len(filtered)} 个模型)")
    norm_df, model_metrics = load_predictions(train_records)
    
    # 划分数据集 (IS / OOS)
    is_norm_df, oos_norm_df = split_is_oos_by_args(norm_df, args)
    if is_norm_df.empty:
        print("错误: IS 期无数据！请检查日期参数。")
        sys.exit(1)

    # Stage 2: 相关性分析
    corr_matrix = correlation_analysis(is_norm_df, args.output_dir, anchor_date)

    # Stage 3: 暴力回测 (基于 IS)
    results_df = brute_force_backtest(
        norm_df=is_norm_df,
        top_k=top_k,
        drop_n=drop_n,
        benchmark=benchmark,
        freq=args.freq,
        min_combo_size=args.min_combo_size,
        max_combo_size=args.max_combo_size,
        output_dir=args.output_dir,
        anchor_date=anchor_date,
        resume=args.resume,
        n_jobs=args.n_jobs,
        use_groups=args.use_groups,
        group_config=args.group_config,
        batch_size=args.batch_size,
    )

    # 导出 Metadata
    metadata_path = os.path.join(args.output_dir, f"run_metadata_{anchor_date}.json")
    is_dates_all = is_norm_df.index.get_level_values("datetime")
    oos_dates_all = oos_norm_df.index.get_level_values("datetime")
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "anchor_date": anchor_date,
            "script_used": "brute_force_ensemble",
            "freq": args.freq,
            "record_file": args.record_file,
            "training_mode": getattr(args, "training_mode", None),
            "is_start_date": str(is_dates_all.min().date()) if not is_dates_all.empty else None,
            "is_end_date": str(is_dates_all.max().date()) if not is_dates_all.empty else None,
            "oos_start_date": str(oos_dates_all.min().date()) if not oos_dates_all.empty else None,
            "oos_end_date": str(oos_dates_all.max().date()) if not oos_dates_all.empty else None,
            "exclude_last_years": args.exclude_last_years,
            "exclude_last_months": args.exclude_last_months,
            "use_groups": args.use_groups,
            "group_config": args.group_config
        }, f, indent=4)
    print(f"\n✅ 元数据已保存: {metadata_path}")
    print("请使用 quantpits/scripts/analyze_ensembles.py 单独进行分析与 OOS 验证。")

    print(f"\n{'='*60}")
    print(f"全部完成！ 耗时结束于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
