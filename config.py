import os

# PATHS
HOME_PATH = os.path.dirname(os.path.abspath(__file__))
ORIG_TRAIN_DATA_PATH = os.path.join(HOME_PATH, 'data/original/train.csv')
ORIG_TEST_DATA_PATH = os.path.join(HOME_PATH, 'data/original/test.csv')

PREPARED_DATA_PATH = os.path.join(HOME_PATH, 'data/prepared')
PREPARED_TRAIN_DATA_PATH = os.path.join(PREPARED_DATA_PATH, 'train.pkl')
PREPARED_TEST_DATA_PATH = os.path.join(PREPARED_DATA_PATH, 'test.pkl')

SAMPLE_SUBMISSION_PATH = os.path.join(HOME_PATH, 'data/original/sample_solution.csv')
SUBMISSION_PATH = os.path.join(HOME_PATH, 'submissions')
OOF_PRED_PATH = os.path.join(HOME_PATH, 'oof_pred')
TEST_PRED_PATH = os.path.join(HOME_PATH, 'test_pred')
MODELS_PATH = os.path.join(HOME_PATH, 'checkpoints')


RENAME_MAP = {
    'ID': 'id',
    'Пол': 'sex',
    'Семья': 'family',
    'Этнос': 'ethnos',
    'Национальность': 'nationality',
    'Религия': 'religion',
    'Образование': 'education',
    'Профессия': 'profession',
    'Вы работаете?': 'employed',
    'Выход на пенсию': 'retired',
    'Прекращение работы по болезни': 'stopping_work_due_to_illness',
    'Сахарный диабет': 'diabetes',
    'Гепатит': 'hepatitis',
    'Онкология': 'oncology',
    'Хроническое заболевание легких': 'chronic_lung_disease',
    'Бронжиальная астма': 'bronchial_asthma',
    'Туберкулез легких ': 'pulmonary_tuberculosis',
    'ВИЧ/СПИД': 'HIV_AIDS',
    'Регулярный прим лекарственных средств': 'regular_medication_intake',
    'Травмы за год': 'injuries_per_year',
    'Переломы': 'fractures',
    'Статус Курения': 'smoking_status', 
    'Возраст курения': 'smoking_age',
    'Сигарет в день': 'cigarettes_a_day',
    'Пассивное курение': 'second_hand_smoke',
    'Частота пасс кур': 'passive_smoking_frequency',
    'Алкоголь': 'alcohol',
    'Возраст алког': 'alcohol_age',
    'Время засыпания': 'sleep_time',
    'Время пробуждения': 'wake_up_time',
    'Сон после обеда': 'sleep_after_dinner',
    'Спорт, клубы': 'sport_clubs',
    'Религия, клубы': 'religion_clubs',
    'Артериальная гипертензия': 'arterial_hypertension',
    'Стенокардия, ИБС, инфаркт миокарда': 'angina_pectoris_ischemic_heart_disease_myocardial_infarction',
    'Сердечная недостаточность': 'heart_failure',
    'Прочие заболевания сердца': 'other_heart_diseases',
    'ОНМК': 'stroke'
}


def get_renamed_cols(columns: list) -> list:
    renamed_cols = []
    for col in columns:
        if col.endswith('_поряд'):
            orig_name = col.split('_поряд')[0]
            renamed_name = RENAME_MAP[orig_name]
            renamed_cols.append(f'{renamed_name}_ordinal')
        else:
            renamed_cols.append(RENAME_MAP[col])
    return renamed_cols

# COLUMNS
CAT_UNORDERED_COLS = get_renamed_cols([
    'Пол', 'Семья', 'Этнос', 'Национальность', 'Религия', 'Образование', 'Профессия', 'Статус Курения',
    'Частота пасс кур', 'Алкоголь', 'Время засыпания', 'Время пробуждения'])


CAT_ORDERED_COLS = get_renamed_cols(['Образование_поряд', 'Статус Курения_поряд', 'Частота пасс кур_поряд'])

BINARY_COLS = get_renamed_cols([
    'Вы работаете?', 'Выход на пенсию', 'Прекращение работы по болезни', 'Сахарный диабет',
    'Гепатит', 'Онкология', 'Хроническое заболевание легких', 'Бронжиальная астма',
    'Туберкулез легких ', 'ВИЧ/СПИД', 'Регулярный прим лекарственных средств', 'Травмы за год', 'Переломы',
    'Пассивное курение', 'Сон после обеда', 'Спорт, клубы', 'Религия, клубы',  
    ])

REAL_COLS = get_renamed_cols([
    'Возраст курения', 'Сигарет в день', 'Возраст алког', 'Время засыпания_поряд', 
    'Время пробуждения_поряд'])

TARGETS = get_renamed_cols([
    'Сердечная недостаточность', 'Стенокардия, ИБС, инфаркт миокарда', 
    'Прочие заболевания сердца', 'ОНМК', 'Артериальная гипертензия'])