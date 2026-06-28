from pathlib import Path
import json, time, shutil

def register_model(cfg, checkpoint_path, normalizer_path=None, metrics_path=None, notes=''):
    registry_dir = Path(cfg['paths'].get('model_registry_dir', 'outputs/model_registry'))
    registry_dir.mkdir(parents=True, exist_ok=True)
    version = time.strftime('model_%Y%m%d_%H%M%S')
    version_dir = registry_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path=Path(checkpoint_path); shutil.copy2(checkpoint_path, version_dir/checkpoint_path.name)
    if normalizer_path and Path(normalizer_path).exists(): shutil.copy2(normalizer_path, version_dir/Path(normalizer_path).name)
    if metrics_path and Path(metrics_path).exists(): shutil.copy2(metrics_path, version_dir/Path(metrics_path).name)
    card={'version':version,'created_at':time.strftime('%Y-%m-%d %H:%M:%S'),'project':cfg.get('project',{}),'model':cfg.get('model',{}),'data':cfg.get('data',{}),'deployment':cfg.get('deployment',{}),'checkpoint':checkpoint_path.name,'normalizer':Path(normalizer_path).name if normalizer_path else None,'metrics':Path(metrics_path).name if metrics_path else None,'notes':notes}
    (version_dir/'model_card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    return version_dir
