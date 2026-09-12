"""Train the four models using the grids and CV procedure in Day_49.ipynb.

Run from Day 50:
python -m house_price_prediction.src.train_notebook
Optional experiment tracking: append --mlflow

Scores reproduce the notebook methodology: preprocessing precedes CV and
hyperparameter selection reuses the same folds. They are not unbiased holdout
scores. The saved regressors accept prepared/scaled matrices, not raw CSVs.
"""
import argparse
import json
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits
from .config import PROJECT_ROOT


def event(name, **values):
    print(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(),
                      'event': name, **values}, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Training sesuai notebook Day 49')
    parser.add_argument('--mlflow', action='store_true', help='Log experiments to local MLflow database')
    args = parser.parse_args()
    source = PROJECT_ROOT / 'data/processed_notebook/prepared.joblib'
    if not source.exists():
        raise SystemExit('Jalankan python -m house_price_prediction.src.preprocessing_notebook --save terlebih dahulu.')
    # Load only your own local artifact, never an untrusted joblib file.
    data = joblib.load(source)
    X, Xt, y = data['X_train_s'], data['X_test_s'], data['y']
    if X.shape != data['X_train'].shape or Xt.shape[1] != X.shape[1] or len(y) != len(X):
        raise ValueError('Shape matriks/target tidak konsisten. Jalankan preprocessing ulang.')
    if not all(np.isfinite(a).all() for a in [X, Xt, np.asarray(y)]):
        raise ValueError('Matriks/target mengandung NaN atau inf.')
    tracking = None
    if args.mlflow:
        try:
            import mlflow as tracking
        except ImportError:
            raise SystemExit('MLflow belum terpasang. Instal dependensi project atau jalankan tanpa --mlflow.')
        tracking.set_tracking_uri('sqlite:///' + (PROJECT_ROOT / 'mlflow.db').as_posix())
        tracking.set_experiment('house-prices-notebook')
    output = PROJECT_ROOT / 'reports/notebook_method'
    models_dir = PROJECT_ROOT / 'models/notebook_method'
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for name in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            event('training_started', model=name, rows=len(X), features=X.shape[1])
            if name == 'LinearRegression':
                model = LinearRegression()
            elif name == 'Ridge':
                search = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=kf).fit(X, y)
                model = Ridge(alpha=search.alpha_)
            elif name == 'Lasso':
                search = LassoCV(alphas=np.logspace(-4, 0, 50), cv=kf, max_iter=20000).fit(X, y)
                model = Lasso(alpha=search.alpha_, max_iter=20000)
            else:
                search = ElasticNetCV(alphas=np.logspace(-4, 0, 30),
                    l1_ratio=[.1,.3,.5,.7,.9,.95,.99,1], cv=kf, max_iter=20000).fit(X,y)
                model = ElasticNet(alpha=search.alpha_, l1_ratio=search.l1_ratio_, max_iter=20000)
            scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
            model.fit(X, y)
            original = np.expm1(np.asarray(y))
            price = np.clip(np.expm1(model.predict(X)), 0, None)
            row = {'model':name, 'alpha':float(model.alpha) if hasattr(model,'alpha') else None,
                   'l1_ratio':float(model.l1_ratio) if hasattr(model,'l1_ratio') else None,
                   'cv_rmse_log_mean':float(-scores.mean()), 'cv_rmse_log_std':float(scores.std()),
                   'train_rmse_price':float(np.sqrt(mean_squared_error(original,price))),
                   'train_mae_price':float(mean_absolute_error(original,price)),
                   'train_r2_price':float(r2_score(original,price))}
            fitted[name] = model
            path = models_dir / (name.lower()+'.joblib')
            joblib.dump(model,path)
            if tracking:
                with tracking.start_run(run_name=name) as run:
                    tracking.log_params({'model':name,'cv_folds':5,'random_state':42,
                        'features':X.shape[1], 'target':'log1p(SalePrice)', **model.get_params()})
                    tracking.set_tag('evaluation_caveat','Preprocessing before CV; tuning reuses folds; notebook reproduction')
                    tracking.log_metrics({key:row[key] for key in ['cv_rmse_log_mean','cv_rmse_log_std',
                        'train_rmse_price','train_mae_price','train_r2_price']})
                    tracking.log_artifact(str(path),artifact_path='models')
                    row['mlflow_run_id'] = run.info.run_id
            rows.append(row)
            event('training_completed', **row)
        # The notebook explicitly uses ElasticNet for submission, regardless of ranking.
        final_model = fitted['ElasticNet']
        pred_price = np.clip(np.expm1(final_model.predict(Xt)), 0, None)
    if not np.isfinite(pred_price).all():
        raise ValueError('Prediksi test tidak finite.')
    comparison = pd.DataFrame(rows).sort_values('cv_rmse_log_mean')
    comparison.to_csv(output/'model_comparison.csv',index=False)
    pd.DataFrame({'Id':data['X_test'].index, 'SalePrice':pred_price}).to_csv(output/'submission.csv',index=False)
    coef = pd.DataFrame({'feature':data['feature_names'],
                        'coef_elasticnet':final_model.coef_, 'coef_ridge':fitted['Ridge'].coef_})
    coef.sort_values('coef_ridge',key=abs,ascending=False).to_csv(output/'coefficients.csv',index=False)
    bundle = {'model':final_model,'scaler':data['scaler'], 'feature_names':data['feature_names'],
              'dropped_multicol':data['dropped_multicol'], 'target_transform':'log1p',
              'input_contract':'scaled matrix in feature_names order; not raw property records',
              'sklearn_version':sklearn.__version__}
    joblib.dump(bundle,models_dir/'elasticnet_bundle.joblib')
    summary = {'final_model':'ElasticNet','train_rows':len(X),'test_rows':len(Xt),
               'features':X.shape[1],'seconds':time.perf_counter()-started,
               'results':rows,'evaluation':'Notebook CV on preprocessed data, not holdout evaluation',
               'serving_status':'Raw-input preprocessing adapter still required; do not replace existing API model'}
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nPerbandingan model (RMSE CV pada skala log):')
    print(comparison[['model','alpha','l1_ratio','cv_rmse_log_mean','cv_rmse_log_std']].to_string(index=False))
    print('\nModel final: ElasticNet (mengikuti notebook)')
    print('Laporan dan submission:',output)
    print('Model:',models_dir)
    print('Training notebook selesai. Integrasi prediksi CSV mentah belum dilakukan.')


if __name__ == '__main__':
    main()
