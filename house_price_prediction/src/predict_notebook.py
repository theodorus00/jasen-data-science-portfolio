"""Raw CSV adapter for the saved notebook ElasticNet model.
Rebuilds imputation statistics and the original category vocabulary from reference
train/test CSVs, without fitting a model or recalculating VIF. New nominal
categories are rejected: the notebook has no learned treatment for them.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import PROJECT_ROOT, TRAIN_PATH, TEST_PATH

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

FEATURES = nominal + ordinal + binary + temporal + numeric
NONE_COLUMNS = ['Alley','MasVnrType','GarageType','MiscFeature','Fence']
MODE_COLUMNS = ['MSZoning','Electrical','Exterior1st','Exterior2nd','SaleType']
ZERO_COLUMNS = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                'BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']


def encode_ordinal(frame):
    df = frame.copy()
    # Ordinal Encoding
    df['LotShape'] = df['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})
    df['Utilities'] = df['Utilities'].map({'ELO':0,'NoSeWa':1,'NoSewr':2,'AllPub':3})
    df['LandSlope'] = df['LandSlope'].map({'Gtl':2,'Mod':1,'Sev':0})
    
    quality_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1, np.nan:0}
    for col in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual',
                'FireplaceQu','GarageQual','GarageCond','PoolQC']:
        df[col] = df[col].map(quality_map)
    
    df['BsmtExposure'] = df['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1, np.nan:0})
    
    bsmt_fin_map = {'GLQ':5,'ALQ':4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':0, np.nan:0}
    df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map)
    
    df['CentralAir'] = df['CentralAir'].map({'N':0,'Y':1})
    df['Functional'] = df['Functional'].map({'Typ':0,'Min1':1,'Min2':2,'Mod':3,
                                                           'Maj1':4,'Maj2':5,'Sev':6,'Sal':7, np.nan:0})
    df['GarageFinish'] = df['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1, np.nan:0})
    df['PavedDrive'] = df['PavedDrive'].map({'N':0,'P':1,'Y':2})
    return df


def validate_raw(frame):
    if not isinstance(frame, pd.DataFrame) or not 1 <= len(frame) <= 2000:
        raise ValueError('Upload 1 to 2,000 rows per batch.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES) - {'Id'})
    if missing or extra:
        raise ValueError(f'Missing columns: {missing}. Unexpected columns: {extra}.')
    df = frame[[col for col in frame.columns if col in FEATURES]].copy()
    for col in FEATURES:
        if df[col].map(lambda v: isinstance(v, (bool,list,dict))).any():
            raise ValueError(f'{col}: invalid value type.')
    for col in numeric + temporal + ['OverallQual','OverallCond','MSSubClass']:
        parsed = pd.to_numeric(df[col], errors='coerce')
        if (df[col].notna() & parsed.isna()).any() or np.isinf(parsed).any():
            raise ValueError(f'{col}: use finite numbers or empty cells.')
        df[col] = parsed
    encoded = encode_ordinal(df)
    for col in ordinal + binary:
        if (df[col].notna() & encoded[col].isna()).any():
            raise ValueError(f'{col}: unknown category. Use original dataset categories.')
    return df


def finish_cleaning(encoded, state):
    df = encoded.copy()
    for col in NONE_COLUMNS:
        df[col] = df[col].fillna('No')
    for col, value in state['modes'].items():
        df[col] = df[col].fillna(value)
    df['Utilities'] = df['Utilities'].fillna(state['utilities_mode'])
    df['LotFrontage'] = df['LotFrontage'].fillna(
        df['Neighborhood'].map(state['frontage_by_neighborhood'])).fillna(state['frontage_fallback'])
    df['HasGarage'] = (df['GarageType'] != 'No').astype(int)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    for col in ZERO_COLUMNS:
        df[col] = df[col].fillna(0)
    return df


def load_notebook_artifact():
    """Load trusted local artifacts and reconstruct the reference preprocessing state."""
    bundle_path = PROJECT_ROOT / 'models/notebook_method/elasticnet_bundle.joblib'
    if not bundle_path.exists():
        raise FileNotFoundError('Run python -m house_price_prediction.src.train_notebook first.')
    bundle = joblib.load(bundle_path)
    train = pd.read_csv(TRAIN_PATH).set_index('Id')
    test = pd.read_csv(TEST_PATH).set_index('Id')
    raw_train, raw_test = validate_raw(train.drop(columns='SalePrice')), validate_raw(test)
    mapped = encode_ordinal(raw_train)
    medians = mapped.groupby('Neighborhood')['LotFrontage'].median()
    state = {
        'modes':{col:mapped[col].mode().iloc[0] for col in MODE_COLUMNS},
        'utilities_mode':mapped['Utilities'].mode().iloc[0],
        'frontage_by_neighborhood':medians,
        'frontage_fallback':mapped['LotFrontage'].fillna(mapped['Neighborhood'].map(medians)).median(),
    }
    reference = pd.concat([finish_cleaning(mapped,state),
                           finish_cleaning(encode_ordinal(raw_test),state)], axis=0)
    state['categories'] = {col:sorted(reference[col].dropna().unique().tolist()) for col in nominal}
    state['bundle'] = bundle
    # Detect a different reference dataset/encoding schema instead of silently reindexing.
    encoded_reference = _encode_features(reference, state)
    feature_names = bundle['feature_names']
    if encoded_reference.columns.tolist() != feature_names:
        raise ValueError('Reference columns differ from training. Restore the original CSVs or rerun preprocessing and training.')
    # Scaler moments also guard against common reference-data edits.
    means = encoded_reference.iloc[:len(train)].mean().to_numpy()
    if not np.allclose(means, bundle['scaler'].mean_, rtol=1e-9, atol=1e-9):
        raise ValueError('Reference data no longer matches the trained scaler. Restore original CSVs or retrain.')
    return state


def _encode_features(cleaned, state):
    df = cleaned.drop(columns=[col for col,_ in state['bundle']['dropped_multicol']]).copy()
    for col in nominal:
        unknown = df[col].notna() & ~df[col].isin(state['categories'][col])
        if unknown.any():
            raise ValueError(f'{col}: unseen categories {df.loc[unknown,col].unique().tolist()[:5]}.')
        df[col] = pd.Categorical(df[col], categories=state['categories'][col])
    encoded = pd.get_dummies(df, columns=nominal, drop_first=True).astype(float)
    if not np.isfinite(encoded.to_numpy()).all():
        raise ValueError('Missing or invalid values remain after notebook preprocessing. Check required property fields.')
    return encoded


def transform_houses(frame, state):
    raw = validate_raw(frame)
    cleaned = finish_cleaning(encode_ordinal(raw), state)
    encoded = _encode_features(cleaned, state)
    # Explicit order from the trained matrix.
    encoded = encoded[state['bundle']['feature_names']]
    return state['bundle']['scaler'].transform(encoded)


def predict_houses(frame, state):
    matrix = transform_houses(frame, state)
    with np.errstate(over='ignore',invalid='ignore'):
        prices = np.maximum(0, np.expm1(state['bundle']['model'].predict(matrix)))
    if not np.isfinite(prices).all():
        raise ValueError('Input produced non-finite predictions. Check feature values.')
    ids = frame['Id'].to_numpy() if 'Id' in frame else np.arange(1, len(frame)+1)
    return pd.DataFrame({'Id':ids, 'SalePrice':prices})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=TEST_PATH)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'reports/notebook_method/raw_predictions.csv')
    args = parser.parse_args()
    predictions = predict_houses(pd.read_csv(args.input), load_notebook_artifact())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False)
    print(f'Saved {len(predictions)} predictions to {args.output}')


if __name__ == '__main__':
    main()
