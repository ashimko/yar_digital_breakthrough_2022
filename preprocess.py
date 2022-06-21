import pandas as pd
import numpy as np
import re
from config import *


def rename_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(columns=RENAME_MAP)
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return data


def add_ordinal_education_feature(data: pd.DataFrame) -> None:
    # образование как порядковый признак
    data['education_ordinal'] = data['education'].str.slice(0, 1).astype(np.int8)


def add_ordinal_smoke_status(data: pd.DataFrame) -> None:
    data['smoking_status_ordinal'] = (
        data['smoking_status'].replace({
            'Никогда не курил(а)': 0,
            'Никогда не курил': 0,
            'Бросил(а)': 1,
            'Курит': 2
        }))


def add_ordinal_smoke_frequency(data: pd.DataFrame) -> None:
    data['passive_smoking_frequency_ordinal'] = (
        data['passive_smoking_frequency'].replace({
            '1-2 раза в неделю': 0,
            '3-6 раз в неделю': 1,
            'не менее 1 раза в день': 2,
            '2-3 раза в день': 3,
            '4 и более раз в день': 4
        })
    )


def add_alcohol_ordinal(data: pd.DataFrame) -> None:
    data['alcohol_ordinal'] = (
        data['alcohol'].replace({
            'никогда не употреблял': 0,
            'ранее употреблял': 1,
            'употребляю в настоящее время': 2
        })
    )


def add_sleep_time_ordinal(data: pd.DataFrame) -> None:
    def _process_sleep_time(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s)
        date = pd.Timestamp(s.iloc[0].date())
        mask = s < (date + pd.Timedelta(hours=12))
        s.loc[mask] = s.loc[mask] + pd.Timedelta(days=1)
        s = (s - date) / pd.Timedelta(hours=1)
        return s

    data['sleep_time_ordinal'] = _process_sleep_time(data['sleep_time'])


def add_wakeup_time_ordinal(data: pd.DataFrame) -> None:
    def _process_wakeup_time(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s)
        date = pd.Timestamp(s.iloc[0].date())
        return (s - date) / pd.Timedelta(hours=1)

    data['wake_up_time_ordinal'] = _process_wakeup_time(data['wake_up_time'])


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    add_ordinal_education_feature(data)
    add_ordinal_smoke_status(data)
    add_ordinal_smoke_frequency(data)
    add_alcohol_ordinal(data)
    add_sleep_time_ordinal(data)
    add_wakeup_time_ordinal(data)
    return data


def cast_types(data: pd.DataFrame) -> pd.DataFrame:
    data[REAL_COLS] = data[REAL_COLS].astype(np.float32)
    data[BINARY_COLS] = data[BINARY_COLS].astype(np.int8)
    data[CAT_ORDERED_COLS] = data[CAT_ORDERED_COLS].fillna(-1).astype(np.int32)
    data[CAT_UNORDERED_COLS] = data[CAT_UNORDERED_COLS].fillna('NA').astype('category')
    return data


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = rename_data(data)
    data = add_features(data)
    data = cast_types(data)
    data = data.set_index('id')
    return data


def main():
    train = pd.read_csv(ORIG_TRAIN_DATA_PATH).drop('ID_y', axis=1)
    train = preprocess(train)
    train.to_pickle(PREPARED_TRAIN_DATA_PATH)

    test = pd.read_csv(ORIG_TEST_DATA_PATH)
    test = preprocess(test)
    test.to_pickle(PREPARED_TEST_DATA_PATH)


if __name__ == '__main__':
    main()