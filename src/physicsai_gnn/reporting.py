from pathlib import Path
import json
import pandas as pd

def _flatten_eval(payload):
    rows=[]
    for batch_idx,item in enumerate(payload):
        for group,metrics_group in item.items():
            for output,metrics in metrics_group.items():
                row={'batch':batch_idx,'group':group,'output':output}; row.update(metrics); rows.append(row)
    return pd.DataFrame(rows)

def generate_validation_report(cfg, eval_json, training_log=None, output_path=None):
    eval_json=Path(eval_json); payload=json.loads(eval_json.read_text(encoding='utf-8'))
    df=_flatten_eval(payload)
    output_path=Path(output_path or (Path(cfg['paths']['output_dir'])/'validation_report.md'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines=[f"# {cfg.get('deployment',{}).get('report_title','CAE Surrogate Validation Report')}", '', '## Model', '', f"- Architecture: `{cfg.get('model',{}).get('name','unknown')}`", f"- Latent dimension: `{cfg.get('model',{}).get('latent_dim','unknown')}`", f"- Local message-passing steps: `{cfg.get('model',{}).get('local_message_passing_steps','n/a')}`", f"- Global transformer steps: `{cfg.get('model',{}).get('global_transformer_steps','n/a')}`", '', '## Evaluation summary', '']
    if not df.empty:
        summary=df.groupby(['group','output'])[['MAE','RMSE','MAPE_%','R2']].mean().reset_index()
        lines.append(summary.to_markdown(index=False))
    else: lines.append('No evaluation rows found.')
    if training_log and Path(training_log).exists():
        log=pd.read_csv(training_log); lines += ['', '## Training summary', '', f"- Epochs completed: {len(log)}", f"- Best validation loss: {log['val_loss'].min():.6g}", f"- Final validation loss: {log['val_loss'].iloc[-1]:.6g}"]
    lines += ['', '## Engineering caution', '', 'This model is a design-space surrogate. Validate against CFD/FEA before engineering release.']
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    csv_path=output_path.with_suffix('.csv'); df.to_csv(csv_path,index=False)
    return output_path,csv_path
