"""Reproduce preprocessing cells 3-12 of Day_49.ipynb.

Run from Day 50:
    python -m house_price_prediction.src.preprocessing_notebook

This module prepares the notebook matrices; it does not train a model or
replace the serving pipeline. Combined train/test encoding and preprocessing
before cross-validation intentionally reproduce the notebook, not a leakage-free
cross-validation estimate. VIF is computed without adding an intercept, as in
the original notebook. Use per-fold fitting for a separate unbiased evaluation.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from threadpoolctl import threadpool_limits
from .data import load_data
from .config import PROJECT_ROOT

nominal = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig',
    'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle',
    'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating',
    'Electrical','GarageType','MiscFeature','SaleType','SaleCondition','Fence']

ordinal = ['LotShape','Utilities','LandSlope','OverallQual','OverallCond','ExterQual',
    'ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual',
    'GarageCond','PavedDrive','PoolQC']

binary = ['CentralAir']
temporal = ['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']

numeric = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
    'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
    'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
    'Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
    '3SsnPorch','ScreenPorch','PoolArea','MiscVal']


def _prepare_data():
    _, _, y_raw, combined, train_mask = load_data()
    combined = combined.copy()
    # Ordinal Encoding
    combined['LotShape'] = combined['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})
    combined['Utilities'] = combined['Utilities'].map({'ELO':0,'NoSeWa':1,'NoSewr':2,'AllPub':3})
    combined['LandSlope'] = combined['LandSlope'].map({'Gtl':2,'Mod':1,'Sev':0})
    
    quality_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1, np.nan:0}
    for col in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual',
                'FireplaceQu','GarageQual','GarageCond','PoolQC']:
        combined[col] = combined[col].map(quality_map)
    
    combined['BsmtExposure'] = combined['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1, np.nan:0})
    
    bsmt_fin_map = {'GLQ':5,'ALQ':4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':0, np.nan:0}
    combined['BsmtFinType1'] = combined['BsmtFinType1'].map(bsmt_fin_map)
    combined['BsmtFinType2'] = combined['BsmtFinType2'].map(bsmt_fin_map)
    
    combined['CentralAir'] = combined['CentralAir'].map({'N':0,'Y':1})
    combined['Functional'] = combined['Functional'].map({'Typ':0,'Min1':1,'Min2':2,'Mod':3,
                                                           'Maj1':4,'Maj2':5,'Sev':6,'Sal':7, np.nan:0})
    combined['GarageFinish'] = combined['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1, np.nan:0})
    combined['PavedDrive'] = combined['PavedDrive'].map({'N':0,'P':1,'Y':2})
    
    # 4a. Nominal yang NaN artinya "fitur tidak ada" -> isi 'No'
    na_means_none = ['Alley','MasVnrType','GarageType','MiscFeature','Fence']
    for col in na_means_none:
        combined[col] = combined[col].fillna('No')
    
    # 4b. Nominal yang NaN-nya benar2 data hilang -> isi modus dari train
    true_missing_nominal = ['MSZoning','Electrical','Exterior1st','Exterior2nd','SaleType']
    for col in true_missing_nominal:
        mode_val = combined.loc[train_mask, col].mode()[0]
        combined[col] = combined[col].fillna(mode_val)
    
    combined['Utilities'] = combined['Utilities'].fillna(combined.loc[train_mask,'Utilities'].mode()[0])
    
    # 4c. LotFrontage -> median per Neighborhood (rumah bertetangga cenderung mirip lebar lotnya)
    neigh_median = combined.loc[train_mask].groupby('Neighborhood')['LotFrontage'].median()
    combined['LotFrontage'] = combined.apply(
        lambda r: neigh_median.get(r['Neighborhood'], np.nan) if pd.isna(r['LotFrontage']) else r['LotFrontage'], axis=1)
    combined['LotFrontage'] = combined['LotFrontage'].fillna(combined.loc[train_mask,'LotFrontage'].median())
    
    # 4d. MasVnrArea kosong -> 0 (tidak ada veneer)
    combined['MasVnrArea'] = combined['MasVnrArea'].fillna(0)
    
    # 4e. GarageYrBlt kosong (tidak ada garasi) -> isi dengan YearBuilt, bukan 0
    #     (mengisi 0 akan membuat outlier ekstrem yang merusak model linear)
    combined['HasGarage'] = (combined['GarageType'] != 'No').astype(int)
    combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(combined['YearBuilt'])
    
    # 4f. Sisa kolom numerik basement/garage yang kosong (rumah tanpa basement/garasi) -> 0
    leftover_zero = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
                      'BsmtHalfBath','GarageCars','GarageArea']
    for col in leftover_zero:
        combined[col] = combined[col].fillna(0)
    
    print('Sisa missing value:', combined.isnull().sum().sum())
    
    continuous_cols = ordinal + binary + ['HasGarage'] + temporal + numeric
    X_corr = combined.loc[train_mask, continuous_cols].astype(float)
    
    corr = X_corr.corr().abs()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    print('Top 10 pasangan fitur paling berkorelasi:')
    print(pairs.head(10))
    
    def compute_vif(df):
        return pd.Series(
            [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
            index=df.columns
        ).sort_values(ascending=False)
    
    X_work = X_corr.copy()
    dropped_multicol = []
    while True:
        vifs = compute_vif(X_work)
        if vifs.iloc[0] > 10 and X_work.shape[1] > 1:
            col = vifs.index[0]
            dropped_multicol.append((col, vifs.iloc[0]))
            X_work = X_work.drop(columns=[col])
        else:
            break
    
    print('Fitur yang dibuang karena VIF > 10 (dibuang satu per satu, VIF dihitung ulang tiap iterasi):')
    for c, v in dropped_multicol:
        print(f'  {c}: VIF awal saat dibuang = {v:.1f}')
    print()
    print('VIF fitur yang tersisa:')
    print(compute_vif(X_work))
    
    combined = combined.drop(columns=[c for c,_ in dropped_multicol])
    print('Kolom yang dibuang:', [c for c,_ in dropped_multicol])
    print('Sisa kolom:', combined.shape[1])
    
    combined_encoded = pd.get_dummies(combined, columns=nominal, drop_first=True)
    print('Shape setelah one-hot encoding:', combined_encoded.shape)
    
    train_mask = combined_encoded['__is_train'] == 1
    X_all = combined_encoded.drop(columns=['__is_train'])
    
    X_train = X_all.loc[train_mask].astype(float)
    X_test = X_all.loc[~train_mask].astype(float)
    y = np.log1p(y_raw.loc[X_train.index])
    
    print('SalePrice skew (asli):', y_raw.skew().round(3))
    print('SalePrice skew (log1p):', y.skew().round(3))
    print('X_train:', X_train.shape, ' X_test:', X_test.shape)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if not np.isfinite(X_train_s).all() or not np.isfinite(X_test_s).all():
        raise ValueError("Preprocessing menghasilkan NaN/inf; periksa data dan mapping kategori.")
    print('Setelah scaling:', X_train_s.shape, X_test_s.shape)
    return {
        'X_train': X_train, 'X_test': X_test,
        'X_train_s': X_train_s, 'X_test_s': X_test_s,
        'y': y, 'y_raw': y_raw.loc[X_train.index],
        'scaler': scaler, 'feature_names': X_train.columns.tolist(),
        'dropped_multicol': dropped_multicol,
        'remaining_vif': compute_vif(X_work),
        'top_correlations': pairs.head(10),
    }


def prepare_data():
    """Return named matrices, target, scaler, feature names, and VIF diagnostics."""
    # Small VIF regressions are faster with one BLAS thread; restore on exit.
    with threadpool_limits(limits=1):
        return _prepare_data()


def main():
    parser = argparse.ArgumentParser(description='Preprocessing sesuai notebook Day 49')
    parser.add_argument('--save', action='store_true', help='Simpan matriks untuk tahap training berikutnya')
    args = parser.parse_args()
    result = prepare_data()
    if args.save:
        destination = PROJECT_ROOT / 'data' / 'processed_notebook'
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, destination / 'prepared.joblib')
        result['remaining_vif'].rename('VIF').to_csv(destination / 'remaining_vif.csv')
        pd.DataFrame(result['dropped_multicol'], columns=['feature', 'vif_at_removal']).to_csv(
            destination / 'dropped_vif.csv', index=False)
        summary = {
            'train_shape': list(result['X_train_s'].shape),
            'test_shape': list(result['X_test_s'].shape),
            'target': 'log1p(SalePrice)',
            'features': result['feature_names'],
            'dropped_features': [name for name, _ in result['dropped_multicol']],
            'method': 'Day_49 notebook reproduction; not fold-isolated preprocessing',
        }
        (destination / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('Hasil disimpan di:', destination)
    print('Preprocessing notebook selesai. Model belum dilatih.')


if __name__ == '__main__':
    main()
