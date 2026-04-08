"""
Run all baselines across all 6 transfer pairs.
Results saved to results/ as JSON.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataset, encode_trajectory
from train import set_seed, DOMAIN_NAMES
from baselines import (
    run_iboat, run_t2vec_knn, run_deep_sad, run_dann,
    run_protonet, run_source_only, run_target_oracle, run_adaptime_knn,
)


def get_kshot_samples(target_train_ds, k, target_domain_id, seed=42):
    """Sample K anomaly + K normal examples from target train set."""
    if k == 0:
        return []
    rng = random.Random(seed)
    anoms = [(c, t, l, s) for c, t, l, s in target_train_ds.samples if l == 1]
    norms = [(c, t, l, s) for c, t, l, s in target_train_ds.samples if l == 0]
    rng.shuffle(anoms)
    rng.shuffle(norms)
    selected = anoms[:k] + norms[:k]

    items = []
    for coords, timestamps, label, subtype in selected:
        tokens = encode_trajectory(coords, timestamps, 128)
        tokens['label'] = torch.tensor(label, dtype=torch.long)
        tokens['domain_id'] = torch.tensor(target_domain_id, dtype=torch.long)
        tokens['subtype'] = subtype
        items.append(tokens)
    return items


def main():
    parser = argparse.ArgumentParser("Run All Baselines")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--k_shots', type=int, nargs='+', default=[0, 5, 10, 20])
    parser.add_argument('--d_model', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = [
        ('porto', 'tdrive'), ('tdrive', 'porto'),
        ('porto', 'geolife'), ('geolife', 'porto'),
        ('tdrive', 'geolife'), ('geolife', 'tdrive'),
    ]
    domain_id_map = {'porto': 0, 'tdrive': 1, 'geolife': 2}

    for src_name, tgt_name in pairs:
        print(f"\n{'='*60}")
        print(f"  Transfer: {src_name} → {tgt_name}")
        print(f"{'='*60}")

        for seed in args.seeds:
            set_seed(seed)

            # Load datasets — use ONLY the specified source domain
            source_names = [src_name]
            source_datasets = []
            for n in source_names:
                try:
                    ds = get_dataset(n, args.data_path, split='train', seed=seed)
                    source_datasets.append(ds)
                except FileNotFoundError:
                    print(f"  Skipping {n} — not found")

            target_train = get_dataset(tgt_name, args.data_path, split='train', seed=seed)
            target_test = get_dataset(tgt_name, args.data_path, split='test', seed=seed)
            tgt_domain_id = domain_id_map[tgt_name]

            for k in args.k_shots:
                kshot = get_kshot_samples(target_train, k, tgt_domain_id, seed)
                prefix = f"{src_name}_to_{tgt_name}_k{k}_s{seed}"

                # 1. IBOAT (K-independent)
                if k == 0:
                    try:
                        r = run_iboat(source_datasets, target_test, args)
                        save_result(args.output_dir, f"bl_iboat_{prefix}", r,
                                    'iboat', src_name, tgt_name, k, seed)
                    except Exception as e:
                        print(f"  IBOAT failed: {e}")

                # 2. t2vec + kNN (K-independent)
                if k == 0:
                    try:
                        r = run_t2vec_knn(source_datasets, target_test, args, device)
                        save_result(args.output_dir, f"bl_t2vec_knn_{prefix}", r,
                                    't2vec_knn', src_name, tgt_name, k, seed)
                    except Exception as e:
                        print(f"  t2vec+kNN failed: {e}")

                # 3. Deep SAD
                try:
                    r = run_deep_sad(source_datasets, target_test, kshot, args, device)
                    save_result(args.output_dir, f"bl_deep_sad_{prefix}", r,
                                'deep_sad', src_name, tgt_name, k, seed)
                except Exception as e:
                    print(f"  Deep SAD failed: {e}")

                # 4. DANN + Transformer
                try:
                    r = run_dann(source_datasets, target_train, target_test, args, device)
                    save_result(args.output_dir, f"bl_dann_{prefix}", r,
                                'dann', src_name, tgt_name, k, seed)
                except Exception as e:
                    print(f"  DANN failed: {e}")

                # 5. ProtoNet
                try:
                    r = run_protonet(source_datasets, target_test, kshot, args, device)
                    save_result(args.output_dir, f"bl_protonet_{prefix}", r,
                                'protonet', src_name, tgt_name, k, seed)
                except Exception as e:
                    print(f"  ProtoNet failed: {e}")

                # 6. Source-only (K-independent)
                if k == 0:
                    try:
                        r = run_source_only(source_datasets, target_test, args, device)
                        save_result(args.output_dir, f"bl_source_only_{prefix}", r,
                                    'source_only', src_name, tgt_name, k, seed)
                    except Exception as e:
                        print(f"  Source-only failed: {e}")

                # 7. Target oracle (K-independent upper bound)
                if k == 0:
                    try:
                        r = run_target_oracle(target_train, target_test, args, device)
                        save_result(args.output_dir, f"bl_target_oracle_{prefix}", r,
                                    'target_oracle', src_name, tgt_name, k, seed)
                    except Exception as e:
                        print(f"  Target oracle failed: {e}")

                # 8. AdapTime + kNN (K-independent)
                if k == 0:
                    try:
                        r = run_adaptime_knn(source_datasets, target_test, args, device)
                        save_result(args.output_dir, f"bl_adaptime_{prefix}", r,
                                    'adaptime_knn', src_name, tgt_name, k, seed)
                    except Exception as e:
                        print(f"  AdapTime failed: {e}")


def save_result(output_dir, name, result, method, src, tgt, k, seed):
    result.update({
        'method': method, 'source': src, 'target': tgt,
        'k_shot': k, 'seed': seed,
    })
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  {method}: AUROC={result['auroc']:.4f} AUPRC={result['auprc']:.4f}")


if __name__ == '__main__':
    main()
