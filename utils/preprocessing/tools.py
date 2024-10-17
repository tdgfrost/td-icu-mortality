import polars as pl
import numpy as np
import h5py

import os
import shutil
import pickle
import sys
import time

pl.Config.set_tbl_cols(10000)
pl.Config.set_tbl_width_chars(10000)
pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_str_lengths(10000)
pl.enable_string_cache()

# ========================================================================== #
# This contains all the processing functions for generating the datasets.
# The functions are sorted in alphabetical order.
# ========================================================================== #


def announce_progress(announcement):
    """
        Print-out of update statements

        :param announcement: String to be printed
        :return: None
    """

    print('\n', '=' * 50, '\n')
    print(' ' * 2, announcement)
    print('\n', '=' * 50, '\n')


def add_demographics_for_sicdb(df, cases):
    """
    Add patientweight, gender, age, and death data
    """
    df = (
        df
        .join(cases.select('CaseID', 'WeightOnAdmission', 'Sex', 'AgeOnAdmission', 'OffsetOfDeath'),
              on='CaseID', how='inner')
        .rename({'WeightOnAdmission': 'patientweight', 'Sex': 'gender', 'AgeOnAdmission': 'age',
                 'OffsetOfDeath': 'deathtime'})
        .with_columns([
            # Change patientweight from grams to kg
            pl.col('patientweight') / 1000,
            # Change gender from 735/736 to 'M'/'F'
            pl.col('gender').replace({735: 0, 736: 1}),
            # `time_until_death` is from the case admission time, so we need to add the offset
            ((pl.col('deathtime') * 1e6).cast(pl.Datetime) - pl.col('starttime')).alias('deathtime')
        ])
    )

    # Convert 1-day, 3-day, 7-day, 14-day, and 28-day mortality into reward labels
    df = (
        df
        .with_columns([
            pl.when(pl.col('deathtime') <= pl.duration(days=day))
            # Represent death as a probability of 1
            .then(pl.lit(1).cast(pl.UInt8).alias(f'{day}-day-died'))
            # Represent survival as a probability of 0
            .otherwise(pl.lit(0).alias(f'{day}-day-died')) for day in [1, 3, 7, 14, 28]
        ])
        .drop('deathtime')
        .unique()
    )

    return df


def add_sofa_to_labels_in_mimic(labels, combined_data):
    sofa_score = combined_data.clone().drop('endtime', 'bolus', 'bolusuom')
    # Pull our SOFA score variables into individual columns
    for feature in ["Blood Gas PaO2", 'FiO2', 'MAP', 'GCS - Eye', 'GCS - Motor', 'GCS - Verbal', 'On ventilation',
                    'Platelets', 'Bilirubin', 'Creatinine']:
        sofa_score = sofa_score.with_columns(
            [pl.when(pl.col('feature') == feature).then(pl.col('valuenum').alias(feature)).otherwise(pl.lit(None))])
    for feature in ['Dopamine', 'Dobutamine', 'Adrenaline', 'Noradrenaline']:
        sofa_score = sofa_score.with_columns(
            [pl.when(pl.col('feature') == feature).then(pl.col('rate').alias(feature)).otherwise(pl.lit(None))])

    # Convert the individual GCS elements to a single GCS score
    sofa_score = (
        sofa_score.with_columns([
            (pl.col('GCS - Eye').max() + pl.col('GCS - Motor').max() + pl.col('GCS - Verbal').max())
            .alias('GCS').over('subject_id', 'starttime')])
        .drop('GCS - Eye', 'GCS - Motor', 'GCS - Verbal', 'valuenum', 'rate', 'feature')
        .unique()
        .sort('subject_id', 'starttime')
    )

    # Focus on rows with SOFA relevant data
    sofa_score = sofa_score.filter(
        pl.col('Blood Gas PaO2').is_not_null() | pl.col('FiO2').is_not_null() | pl.col('MAP').is_not_null()
        | pl.col('GCS').is_not_null() | pl.col('On ventilation').is_not_null() | pl.col('Platelets').is_not_null()
        | pl.col('Bilirubin').is_not_null() | pl.col('Creatinine').is_not_null() | pl.col('Dopamine').is_not_null()
        | pl.col('Dobutamine').is_not_null() | pl.col('Adrenaline').is_not_null() | pl.col('Noradrenaline').is_not_null())

    # Fill in missing patientweight values (as we need to convert our rates to mcg/kg/min)
    sofa_score = sofa_score.sort('subject_id', 'starttime').with_columns([
        pl.col('patientweight')
        .fill_null(strategy='forward')
        .fill_null(strategy='backward')
        .fill_null(value=70).over('subject_id')])

    # Convert the rates to mcg/kg/min
    sofa_score = (
        sofa_score.with_columns([
            pl.when(pl.col('rateuom') == 'mg/minute')
            .then(pl.col('Dopamine', 'Dobutamine', 'Noradrenaline') * 1000 / pl.col('patientweight'))
            .otherwise(pl.lit(None))
        ])
        .with_columns([
            pl.when(pl.col('rateuom') == 'mcg/minute')
            .then(pl.col('Adrenaline') / pl.col('patientweight'))
            .otherwise(pl.lit(None))
        ])
    )

    # Change FiO2 to decimal rather than percentage
    sofa_score = sofa_score.with_columns([pl.col('FiO2') / 100])

    # Use a rolling window to get the "worst" values for the last 24 hours at each timepoint
    sofa_score = (
        sofa_score
        .sort(by=['subject_id', 'starttime'])
        # Fill nulls for drugs
        .with_columns([
            pl.col(feature).fill_null(strategy='forward').fill_null(value=0).over('subject_id')
            for feature in ['Dopamine', 'Dobutamine', 'Adrenaline', 'Noradrenaline']
        ])
        .rolling(index_column='starttime', period='24h', group_by='subject_id')
        .agg(pl.col('Blood Gas PaO2').min(), pl.col('FiO2').max(), pl.col('MAP').min(), pl.col('GCS').min(),
             pl.col('On ventilation').max(), pl.col('Platelets').min(), pl.col('Bilirubin').max(),
             pl.col('Creatinine').max(), pl.col('Dopamine').max(), pl.col('Dobutamine').max(),
             pl.col('Adrenaline').max(), pl.col('Noradrenaline').max())
        .with_columns([pl.col('On ventilation').cast(pl.Boolean).fill_null(value=False)])
        # Fill nulls so we can get SOFA scores everywhere
        # .with_columns([
            # pl.col(feature).fill_null(strategy='forward').over('subject_id')
            # for feature in ['Blood Gas PaO2', 'FiO2', 'MAP', 'GCS', 'Platelets', 'Bilirubin', 'Creatinine']
        # ])
    )

    # Now to convert to SOFA
    v_severe_resp = (pl.col('Blood Gas PaO2') / pl.col('FiO2') < 100) & (pl.col('On ventilation'))
    severe_resp = (pl.col('Blood Gas PaO2') / pl.col('FiO2') < 200) & (pl.col('On ventilation'))
    mod_resp = (pl.col('Blood Gas PaO2') / pl.col('FiO2') < 200) & (~pl.col('On ventilation'))
    mild_resp = pl.col('Blood Gas PaO2') / pl.col('FiO2') < 300
    v_mild_resp = pl.col('Blood Gas PaO2') / pl.col('FiO2') < 400
    resp_null = pl.col('Blood Gas PaO2').is_null() | pl.col('FiO2').is_null()

    v_v_low_platelets = pl.col('Platelets') < 20
    v_low_platelets = pl.col('Platelets') < 50
    low_platelets = pl.col('Platelets') < 100
    mild_platelets = pl.col('Platelets') < 150
    platelets_null = pl.col('Platelets').is_null()

    v_low_GCS = pl.col('GCS') < 6
    low_GCS = pl.col('GCS') < 10
    mod_GCS = pl.col('GCS') < 13
    mild_GCS = pl.col('GCS') < 15
    GCS_null = pl.col('GCS').is_null()

    v_high_bili = pl.col('Bilirubin') > 204
    high_bili = pl.col('Bilirubin') >= 102
    mod_bili = pl.col('Bilirubin') >= 33
    mild_bili = pl.col('Bilirubin') >= 20
    bili_null = pl.col('Bilirubin').is_null()

    high_pressors = (pl.col('Dopamine') > 15) | (pl.col('Adrenaline') > 0.1) | (pl.col('Noradrenaline') > 0.1)
    med_pressors = (pl.col('Dopamine') > 5) | (pl.col('Adrenaline') > 0) | (pl.col('Noradrenaline') > 0)
    low_pressors = (pl.col('Dopamine') > 0) | (pl.col('Dobutamine') > 0)
    hypotensive = pl.col('MAP') < 70
    pressors_null = pl.col('MAP').is_null()

    v_high_cr = pl.col('Creatinine') > 440
    high_cr = pl.col('Creatinine') >= 300
    mod_cr = pl.col('Creatinine') >= 171
    mild_cr = pl.col('Creatinine') >= 110
    cr_null = pl.col('Creatinine').is_null()

    sofa_score = (
        sofa_score.with_columns([
            pl.when(v_severe_resp).then(4).when(severe_resp).then(3).when(mod_resp).then(2).when(mild_resp).then(2)
            .when(v_mild_resp).then(1).when(resp_null).then(None).otherwise(0).alias('Resp_1'),

            pl.when(v_v_low_platelets).then(4).when(v_low_platelets).then(3).when(low_platelets).then(2)
            .when(mild_platelets).then(1).when(platelets_null).then(None).otherwise(0).alias('Platelets_2'),

            pl.when(v_low_GCS).then(4).when(low_GCS).then(3).when(mod_GCS).then(2).when(mild_GCS).then(1)
            .when(GCS_null).then(None).otherwise(0).alias('GCS_3'),

            pl.when(v_high_bili).then(4).when(high_bili).then(3).when(mod_bili).then(2).when(mild_bili).then(1)
            .when(bili_null).then(None).otherwise(0).alias('Bilirubin_4'),

            pl.when(high_pressors).then(4).when(med_pressors).then(3).when(low_pressors).then(2)
            .when(hypotensive).then(1).when(pressors_null).then(None).otherwise(0).alias('Pressors_5'),

            pl.when(v_high_cr).then(4).when(high_cr).then(3).when(mod_cr).then(2).when(mild_cr).then(1)
            .when(cr_null).then(None).otherwise(0).alias('Creatinine_6')
        ])
        .select('subject_id', 'starttime', (pl.col('Resp_1') + pl.col('Platelets_2') + pl.col('GCS_3')
                                            + pl.col('Bilirubin_4') + pl.col('Pressors_5') + pl.col('Creatinine_6')
                                            ).alias('SOFA'))
        .filter(pl.col('SOFA').is_not_null())
        # Standardise to 0-1 range, assuming a minimum SOFA score of 0
        .with_columns([
            pl.col('SOFA') / pl.col('SOFA').max()
        ])
        .unique()
    )

    # Join into our labels DF
    labels = labels.join(sofa_score, left_on=['subject_id', 'labeltime'], right_on=['subject_id', 'starttime'],
                         how='left')

    # Ditch the vital signs from the combined_data DF
    combined_data = combined_data.filter(~pl.col('feature').is_in(['FiO2', 'MAP', 'GCS - Eye', 'GCS - Motor',
                                                                   'GCS - Verbal', 'On ventilation']))

    return labels, combined_data


def add_zero_rates(df):
    """
    Rates that are stopped need to be explicitly set to zero.
    """
    # Find the rows that aren't contiguous
    criterion_1 = pl.col('endtime') != pl.col('starttime').shift(-1).over(['subject_id', 'label'])
    criterion_2 = pl.col('starttime').shift(-1).is_null().over(['subject_id', 'label'])
    criteria = criterion_1 | criterion_2

    # Create a dedicated "zero rates" dataframe
    zero_rates = (
        df
        .filter(pl.col('rate').is_not_null())
        .sort(by=['subject_id', 'label', 'starttime', 'endtime'])
        .select([pl.exclude(['starttime', 'endtime', 'rate']),

                 # Find rows where insulin stopped
                 # - set the endtime of the last real rate to the starttime for the 'stopped' row
                 pl.when(criteria)
                .then(pl.col('endtime').alias('starttime'))
                .otherwise(pl.col('starttime')),

                 # - set the starttime of the next real rate to the endtime for the 'restart' row - this may be null
                 #  if no further insulin is restarted
                 pl.when(criteria)
                .then(pl.col('starttime').shift(-1).over(['subject_id', 'label']).alias('endtime'))
                .otherwise(pl.col('endtime')),

                 # - set rate to zero
                 pl.when(criteria)
                .then(pl.lit(0.0).alias('rate'))
                .otherwise(pl.col('rate'))
                 ])
        .select(df.columns)
        .filter(pl.col('rate') == 0.0)

    )

    df = pl.concat([df, zero_rates]).sort(by=['subject_id', 'starttime', 'endtime'])

    return df


def change_antibiotic_doses_for_mimic(df):
    """
    Remove the dose value for antibiotics - just treat as binary event (they are on 'x' antibiotic or not).

    N.B. we will also include IVIG in this list - having some difficulties with doses etc.
    """
    antibiotics = get_antibiotic_names()
    current_labels = df.select('label').unique().to_series().to_list()

    for antibiotic in antibiotics:
        if antibiotic in current_labels:
            criteria = pl.col('label') == antibiotic
            df = (
                df
                .with_columns([
                    # Set bolus to 1.0 for antibiotics
                    pl.when(criteria)
                    .then(pl.lit(1.0).alias('amount'))
                    .otherwise(pl.col('amount')),

                    pl.when(criteria)
                    .then(pl.lit('dose').cast(pl.Categorical('lexical')).alias('amountuom'))
                    .otherwise(pl.col('amountuom')),

                    # Set endtime to null
                    pl.when(criteria)
                    .then(pl.lit(None).alias('endtime'))
                    .otherwise(pl.col('endtime')),

                    # Set ordercategorydescription to "Bolus"
                    pl.when(criteria)
                    .then(pl.lit('Bolus').cast(pl.Categorical('lexical')).alias('ordercategorydescription'))
                    .otherwise(pl.col('ordercategorydescription'))
                ])
            )
    return df


def change_antibiotic_doses_for_sicdb(df):
    """
    Remove the dose value for antibiotics - just treat as binary event (they are on 'x' antibiotic or not).

    - N.B. For SICDB, I actually think they are already recorded as boluses. But this is just to cover any exceptions.
    """
    antibiotics = get_antibiotic_names()
    current_labels = df.select('label').unique().to_series().to_list()

    for antibiotic in antibiotics:
        if antibiotic in current_labels:
            criteria = pl.col('label') == antibiotic
            df = (
                df
                .with_columns([
                    # Set bolus to 1.0 for antibiotics
                    pl.when(criteria)
                    .then(pl.lit(1.0).alias('amount'))
                    .otherwise(pl.col('amount')),

                    pl.when(criteria)
                    .then(pl.lit('dose').cast(pl.Categorical('lexical')).alias('valueuom'))
                    .otherwise(pl.col('valueuom')),

                    # Set endtime to null
                    pl.when(criteria)
                    .then(pl.lit(None).alias('endtime'))
                    .otherwise(pl.col('endtime')),

                    # Set IsSingleDose to 1
                    pl.when(criteria)
                    .then(pl.lit(1).alias('IsSingleDose'))
                    .otherwise(pl.col('IsSingleDose')),

                    # Set rate to null
                    pl.when(criteria)
                    .then(pl.lit(None).alias('rate'))
                    .otherwise(pl.col('rate'))
                ])
            )
    return df


def change_drug_units(df, variables, unit_str, old_albumin):
    for variable in variables.keys():
        default_unit = variables[variable]['label']
        is_default_unit = pl.col(unit_str) == default_unit
        is_variable = pl.col('label') == variable

        for unit in variables[variable]['convert'].keys():
            is_current_unit = pl.col(unit_str) == unit
            criteria = is_variable & is_current_unit

            df = (
                df.with_columns([
                    pl.when(criteria)
                    .then(pl.col('amount') * variables[variable]['convert'][unit])
                    .otherwise(pl.col('amount')),

                    pl.when(criteria)
                    .then(pl.lit(default_unit).cast(pl.Categorical('lexical')).alias(unit_str))
                    .otherwise(pl.col(unit_str))
                ])
            )

        df = df.filter((is_variable & is_default_unit) | ~is_variable)

    df = (
        df.with_columns([pl.col('label')
                        .cast(pl.Utf8)
                        .replace(old_albumin, 'Human Albumin Solution')
                        .cast(pl.Categorical('lexical'))])
    )

    return df


def change_drug_units_for_mimic(df):
    """
    Where there is inconsistency, group all drug units into a single unit.
    """

    variables = {
        'Adrenaline':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Cisatracurium':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Dexmedetomidine':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Fentanyl':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Furosemide':
            {'convert': {'mg': 1}, 'label': 'mg'},
        'Human Albumin Solution 25%':
            {'convert': {'ml': 0.25}, 'label': 'grams'},  # this is a special case where we wish to convert to grams
        'Ketamine':
            {'convert': {'mcg': 1 / 1000, 'mg': 1}, 'label': 'mg'},
        'Labetalol':
            {'convert': {'mg': 1}, 'label': 'mg'},
        'Levetiracetam':
            {'convert': {'grams': 1000, 'mcg': 1 / 1000, 'mg': 1}, 'label': 'mg'},
        'Phenytoin':
            {'convert': {'grams': 1000, 'mg': 1}, 'label': 'mg'},
        'Propofol':
            {'convert': {'mg': 1, 'mcg': 1 / 1000}, 'label': 'mg'},
        'Rocuronium':
            {'convert': {'mcg': 1 / 1000, 'mg': 1}, 'label': 'mg'},
        'Sodium Bicarbonate 8.4%':
            {'convert': {'ml': 1, 'mEq': 1}, 'label': 'ml'},
    }

    df = change_drug_units(df, variables, 'amountuom', 'Human Albumin Solution 25%')

    return df


def change_drug_units_for_sicdb(df):
    """
    Where there is inconsistency, group all drug units into a single unit.
    """

    variables = {
        'Adrenaline':
            {'convert': {'g': 1000000}, 'label': 'mcg'},
        'Alteplase':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Amiodarone':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Cisatracurium':
            {'convert': {'g': 1000000}, 'label': 'mcg'},
        'Dexmedetomidine':
            {'convert': {'g': 1000000}, 'label': 'mcg'},
        'Dobutamine':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Dopamine':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Fentanyl':
            {'convert': {'g': 1000000}, 'label': 'mcg'},
        'Furosemide':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Human Albumin Solution 20%':
            {'convert': {'ml': 0.2}, 'label': 'grams'},  # this is a special case where we wish to convert to grams
        'Ketamine':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Labetalol':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Levetiracetam':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Lorazepam':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Midazolam':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Milrinone':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Morphine':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Nitroglycerin':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Nitroprusside':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Noradrenaline':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Phenytoin':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Propofol':
            {'convert': {'g': 1000}, 'label': 'mg'},
        'Rocuronium':
            {'convert': {'g': 1000}, 'label': 'mg'},
    }

    df = change_drug_units(df, variables, 'valueuom', 'Human Albumin Solution 25%')

    return df


def change_lab_units(df, variables):
    for col in variables.keys():
        df = (
            df.with_columns([
                pl.when(pl.col('label') == col)
                .then(pl.col('valuenum') * variables[col]['valuenum'])
                .otherwise(pl.col('valuenum')),

                pl.when(pl.col('label') == col)
                .then(pl.lit(variables[col]['valueuom']).cast(pl.Categorical('lexical')).alias('valueuom'))
                .otherwise(pl.col('valueuom'))
            ])
        )

    return df


def change_lab_units_for_mimic(df):
    """
    Goal is to change units to standardised units.
    """
    variables = {'Albumin':  # from g/dL to g/L
                     {'valuenum': 10, 'valueuom': 'g/L'},

                 'Bedside Glucose':  # from mg/dL to mmol/L
                     {'valuenum': 1 / 18, 'valueuom': 'mmol/L'},

                 'Glucose':  # from mg/dL to mmol/L
                     {'valuenum': 1 / 18, 'valueuom': 'mmol/L'},

                 'Bilirubin':  # from mg/dL to µmol/L
                     {'valuenum': 17.1, 'valueuom': 'µmol/L'},

                 'Blood Gas pO2':  # from mmHg to kPa
                     {'valuenum': 0.133322, 'valueuom': 'kPa'},

                 'Blood Gas pCO2':  # from mmHg to kPa
                     {'valuenum': 0.133322, 'valueuom': 'kPa'},

                 'Calcium':
                     {'valuenum': 0.2495, 'valueuom': 'mmol/L'},

                 'Creatinine':  # from mg/dL to µmol/L
                     {'valuenum': 88.42, 'valueuom': 'µmol/L'},

                 'Haemoglobin':  # from g/dL to g/L
                     {'valuenum': 10, 'valueuom': 'g/L'},

                 'Urea':  # from mg/dL to mmol/L
                     {'valuenum': 0.357, 'valueuom': 'mmol/L'}}

    df = change_lab_units(df, variables)

    return df


def change_lab_units_for_sicdb(df):
    """
    Goal is to change units to standardised units.
    """
    variables = {'Albumin':  # from g/dL to g/L
                     {'valuenum': 10, 'valueuom': 'g/L'},

                 'Bedside Glucose':  # from mg/dL to mmol/L
                     {'valuenum': 1 / 18, 'valueuom': 'mmol/L'},

                 'Bilirubin':  # from mg/dL to µmol/L
                     {'valuenum': 17.1, 'valueuom': 'µmol/L'},

                 'Blood Gas pCO2':  # from mmHg to kPa
                     {'valuenum': 0.133322, 'valueuom': 'kPa'},

                 'Blood Gas pO2':  # from mmHg to kPa
                     {'valuenum': 0.133322, 'valueuom': 'kPa'},

                 'Creatinine':  # from mg/dL to µmol/L
                     {'valuenum': 88.42, 'valueuom': 'µmol/L'},

                 'CRP':  # from mg/dL to mg/L
                     {'valuenum': 10, 'valueuom': 'mg/L'},

                 'Glucose':  # from mg/dL to mmol/L
                     {'valuenum': 1 / 18, 'valueuom': 'mmol/L'},

                 'Haemoglobin':  # from g/dL to g/L
                     {'valuenum': 10, 'valueuom': 'g/L'},

                 'Platelets':  # from g/L to K/µL
                     {'valuenum': 1, 'valueuom': 'K/uL'},

                 'Troponin - T':  # from ng/L to ng/mL
                     {'valuenum': 1 / 1000, 'valueuom': 'ng/mL'},

                 'Urea':  # from mg/dL to mmol/L
                     {'valuenum': 0.357, 'valueuom': 'mmol/L'},

                 'WBC':  # from g/L to K/µL
                     {'valuenum': 1, 'valueuom': 'K/uL'}}

    df = change_lab_units(df, variables)

    return df


def clean_combined_data(combined_data, admissions, train_patient_ids, data_option: str):
    cleaning_functions = get_cleaning_functions(data_option)

    combined_data = cleaning_functions['convert_offset_to_datetime'](combined_data)
    combined_data = cleaning_functions['change_lab_units'](combined_data)
    combined_data = cleaning_functions['change_antibiotic_doses'](combined_data)
    combined_data = cleaning_functions['change_drug_units'](combined_data)
    combined_data = cleaning_functions['change_basal_bolus'](combined_data)
    combined_data = cleaning_functions['merge_overlapping_rates'](combined_data)
    combined_data = cleaning_functions['add_zero_rates'](combined_data)
    combined_data = cleaning_functions['add_demographics'](combined_data, admissions.collect())
    combined_data = cleaning_functions['remove_outliers'](combined_data, train_patient_ids)

    if data_option == 'mimic':
        combined_data = (
            combined_data
            .select(['subject_id', 'label', 'starttime', 'endtime', 'valuenum', 'valueuom', 'bolus', 'bolusuom', 'rate',
                     'rateuom', 'patientweight']))

        combined_data = combined_data.rename({'label': 'feature'}).unique().sort(
            by=['subject_id', 'starttime', 'endtime'])

    elif data_option == 'sicdb':
        combined_data = (
            combined_data
            .select(
                ['subject_id', 'CaseID', 'label', 'starttime', 'endtime', 'valuenum', 'valueuom', 'bolus', 'bolusuom',
                 'rate', 'rateuom', 'patientweight', 'gender', 'age', '1-day-died', '3-day-died',
                 '7-day-died', '14-day-died', '28-day-died'])
        )

        combined_data = combined_data.rename({'label': 'feature'}).unique().sort(
            by=['subject_id', 'CaseID', 'starttime', 'endtime'])

    else:
        raise ValueError(f'Invalid data option: {data_option}')

    return combined_data


def convert_dataframe_to_hdf5(data_option: str, context_length: int = 400):
    if data_option == 'mimic':
        iterate_groups = ['train', 'val', 'test']
    elif data_option == 'sicdb':
        iterate_groups = ['test']
    else:
        raise ValueError(f'Invalid data option: {data_option}')

    # Load the features.txt and label_features.txt files
    with open('./data/features.txt', 'r') as f:
        features = f.read().splitlines()
        f.close()

    with open('./data/label_features.txt', 'r') as f:
        label_features = f.read().splitlines()
        f.close()

    for segment in iterate_groups:
        # Define our target paths
        dataframe_dir = f'./data/{data_option}/{segment}/dataframe_{segment}'
        array_dir = f'./data/{data_option}/{segment}'
        h5_array_path = os.path.join(array_dir, f'h5_array_{segment}.hdf5')

        # Create the directories if necessary
        os.makedirs(array_dir, exist_ok=True)

        # Create the hdf5 binary
        create_hdf5_array(h5_array_path, dataframe_dir, features, label_features, context_length)


def convert_sicdb_offset_to_datetime(df):
    """
    Convert the offset to a datetime object.
    Note that the actual year/month/day is not important, just the relative time intervals.
    """
    df = (
        df
        .with_columns([
            (pl.col('starttime') * 1e6).cast(pl.Datetime),  # can do pl.Datetime if needed
            (pl.col('endtime') * 1e6).cast(pl.Datetime)
        ])
    )
    return df


def create_final_dataframe(data_option: str, encoded_input_data, labels, encodings, train_patient_ids, val_patient_ids,
                           test_patient_ids, sorting_columns, grouping_columns, context_length):
    """
    This is the final pipeline for taking the encoded data and labels, and creating dedicated input data
    for the model.

    N.B. We are creating a "sliding window" of data (with measurements being repeated across potentially
    many rows), which means the data is LARGE even with .parquet compression, and may be slow to complete.
    """
    if data_option == 'mimic':
        iterate_groups = [[train_patient_ids, 'train'], [val_patient_ids, 'val'], [test_patient_ids, 'test']]
    elif data_option == 'sicdb':
        iterate_groups = [[test_patient_ids, 'test']]
    else:
        raise ValueError(f'Invalid data option: {data_option}')

    # Iterate through each data segment (train/val/test for mimic, just test for sicdb)
    for patient_ids, group in iterate_groups:
        print(f'Processing {group}...')
        path = f'./data/{data_option}/{group}'
        dataframe_path = os.path.join(path, f'dataframe_{group}')
        os.makedirs(dataframe_path, exist_ok=True)

        chunk_size = 100  # number of patient ids to process at once
        chunk_size = len(patient_ids) // chunk_size + 1

        # Iterate through each chunk of patient ids
        for idx, ids_chunk in progress_bar(np.array_split(patient_ids, chunk_size), with_time=True):

            # Start by creating our base "input data", i.e., for each label, we want to find all eligible measurements
            # within the inclusion window (e.g., the last 7 days of measurements)
            temp = (
                # Join the input data and the labels together
                encoded_input_data
                .filter(pl.col('subject_id').is_in(ids_chunk))
                .join(labels.drop('label', strict=False), on=['subject_id'], how='inner')
                # Filter the measurements to the inclusion period for each label
                .filter(
                    (pl.col('featuretime') >= pl.col('start_inclusion'))
                    & (pl.col('featuretime') <= pl.col('end_inclusion')))
                .drop('start_inclusion')
                # Select the desired columns
                .select(sorting_columns + [
                    (pl.col('end_inclusion') - pl.col('featuretime')).dt.total_minutes().cast(pl.Int16).alias(
                        'featuretime'),
                    'value', 'encoded_feature',
                    # For all our labels, just group these together into a Polars Struct for simplicity
                    pl.struct(
                        pl.col('labeltime_next', '1-day-died', '3-day-died',
                               '7-day-died', '14-day-died', '28-day-died', 'SOFA', 'age', 'gender',
                               'patientweight')).alias('targets')
                ])
                .collect()
                .lazy()
            )

            # As we are limited to 400 measurements in a 7-day window, we want to make sure all current drug infusions
            # are included in the input data. The risk is that in edge cases where we have LOADS of recent measurements,
            # ongoing drug infusions might get missed out as a result.
            current_drug_infusions = (
                temp
                .filter(pl.col('encoded_feature').is_in(encodings['drug_names']))
                .rename({'encoded_feature': 'current_drug_feature', 'value': 'current_drug_value'})
                .sort(by=sorting_columns + ['featuretime'])
                # Following is equivalent to group_by, but is MUCH faster than using group_by directly
                .select(pl.all().first().over(grouping_columns + ['current_drug_feature']))
                .unique()
                # Filter out drugs that have been stopped (i.e., value = 0)
                .filter(pl.col('current_drug_value') > 0)
                # Fill in our delta_time/delta_value and time columns with 0
                .with_columns([pl.lit(0).cast(pl.Int16).alias('current_drug_time'),
                               pl.lit(0).cast(pl.Int16).alias('current_drug_time_delta'),
                               pl.lit(0.).alias('current_drug_value_delta')])
                .drop('featuretime')
                .group_by(grouping_columns)
                .agg(pl.all())
            )

            # Next, we will identify all historic drug rates and all other measurements.
            historic_events = temp.head(0)
            # Iterate through each feature, one at a time
            for feature in encodings['all_features']:
                historic_events = (
                    # Add the current feature to the historic events
                    pl.concat([historic_events,
                               temp
                              .filter(pl.col('encoded_feature') == feature)
                              # Sorted from new to old
                              .sort(by=sorting_columns + ['featuretime'])
                              .with_columns([
                                   # Get the 'rank' of each measurement
                                   # i.e., the number of times the feature has appeared
                                   # (we want to prioritise "unseen" features before repeat features)
                                   pl.col('encoded_feature').cum_count().over(sorting_columns).alias('feature_rank'),

                                   # Calculate the delta_value and delta_time for each feature
                                   # (excluding bolus/event values)
                                   pl.when(feature in encodings['lab_names'] + encodings['drug_names'])
                                   .then((pl.col('featuretime').shift(-1).over(sorting_columns) - pl.col('featuretime'))
                                         .cast(pl.Int16).alias('delta_time'))
                                   .otherwise(pl.lit(None)),

                                   pl.when(feature in encodings['lab_names'] + encodings['drug_names'])
                                   .then((pl.col('value') - pl.col('value').shift(-1).over(sorting_columns))
                                         .cast(pl.Float64).alias('delta_value'))
                                   .otherwise(pl.lit(None))
                               ])
                               ], how='diagonal')  # diagonal allows concat with missing columns
                )

            # Sort all historic measurements by 1) the feature rank, and then 2) the feature time
            historic_events = (
                historic_events
                .sort(sorting_columns + ['feature_rank', 'featuretime'])
                .group_by(grouping_columns)
                .agg(pl.all())
            )

            # Add back in our "current drug" measurements
            target_data = (
                historic_events
                .join(current_drug_infusions, on=grouping_columns, how='full', coalesce=True)
            )

            #
            target_data = (
                target_data
                .with_columns([
                    # Use if-then to avoid concatenating nulls unnecessarily
                    # - if one col is empty, use the other col on its own
                    pl.when(pl.col(first_col).is_null())
                    .then(pl.col(second_col))
                    # - otherwise, if the other col is empty, use the first col on its own
                    .otherwise(pl.when(pl.col(second_col).is_null())
                               .then(pl.col(first_col))
                               # - otherwise, concatenate the two
                               .otherwise(pl.concat_list(pl.col(first_col), pl.col(second_col)))
                               ).alias(second_col)

                    for first_col, second_col in [['current_drug_time', 'featuretime'],
                                                  ['current_drug_value', 'value'],
                                                  ['current_drug_feature', 'encoded_feature'],
                                                  ['current_drug_time_delta', 'delta_time'],
                                                  ['current_drug_value_delta', 'delta_value']]
                ])
                .drop('current_drug_feature', 'current_drug_value', 'current_drug_time', 'current_drug_time_delta',
                      'current_drug_value_delta', 'feature_rank')
            )

            input_feature_columns = ['featuretime', 'encoded_feature', 'value', 'delta_time', 'delta_value']

            # Now for each label, filter to just 397 measurements (i.e., 400 - 3 for age/gender/weight),
            # with nulls added if necessary
            target_data = (
                target_data
                .with_columns([
                    pl.col(col)
                    # We do (max_context_window - 3) because we will be adding age/gender/weight at the start
                    .list.concat([None for _ in range(context_length - 3)])
                    .list.slice(0, context_length - 3)
                    for col in input_feature_columns
                ])
            )

            # Merge gender/age/weight into the lab/event columns
            target_data = (
                target_data
                .unnest('targets')
                # - concat [age, gender, patientweight] to the start of value
                .with_columns([pl.concat_list(pl.concat_list(pl.col('age').cast(pl.Float64),
                                                             pl.col('gender').cast(pl.Float64),
                                                             pl.col('patientweight').cast(pl.Float64)
                                                             # pl.col('minutes_since_admission').cast(pl.Float64)
                                                             ),
                                              pl.col('value'))
                              .alias('value')])
                .drop('age', 'gender', 'patientweight')  # 'minutes_since_admission')
                # - concat [0, 0, 0] to the start of featuretime
                .with_columns([pl.concat_list(pl.lit([np.int16(0) for _ in range(3)]),
                                              pl.col('featuretime'))
                              .alias('featuretime')])
                # - concat encoded feature values for [age, gender, patientweight] to the start of feature
                .with_columns([pl.concat_list(pl.lit([encodings['age'], encodings['gender'], encodings['weight']
                                                      ]),
                                              pl.col('encoded_feature'))
                              .alias('encoded_feature')])
                # - concat [Null, Null, Null] to the start of delta_time and delta_value
                .with_columns([pl.concat_list(pl.lit([None for _ in range(3)]).cast(pl.List(pl.Int16)),
                                              pl.col('delta_time')).alias('delta_time'),

                               pl.concat_list(pl.lit([None for _ in range(3)]),
                                              pl.col('delta_value')).alias('delta_value')])
            )

            # For dtype efficiency, change some of the nulls to -1 (and NaN for the rest)
            target_data = (
                target_data
                .with_columns([
                    pl.col('featuretime').list.eval(pl.element().fill_null(-1)),
                    pl.col('value').list.eval(pl.element().fill_null(np.nan)),
                    pl.col('encoded_feature').list.eval(pl.element().fill_null(-1)),
                    pl.col('delta_time').list.eval(pl.element().fill_null(-1)),
                    pl.col('delta_value').list.eval(pl.element().fill_null(np.nan))
                ])
                .collect()
                .lazy()
            )

            # We need to then get our "next state", defined as 24hrs from now (same as our viewing window)
            has_next_state = target_data.filter(pl.col('labeltime_next').is_not_null())

            has_next_state = (
                has_next_state
                .join(target_data.select(sorting_columns + input_feature_columns),
                      left_on=sorting_columns[:-1] + ['labeltime_next'],
                      right_on=sorting_columns, how='inner', suffix='_next')
                .rename({f'{col}_next': f'next_{col}' for col in ['labeltime'] + input_feature_columns})
            )

            no_next_state = target_data.filter(pl.col('labeltime_next').is_null())

            no_next_state = (
                no_next_state
                .with_columns([
                    pl.lit(None).alias('next_' + col) for col in input_feature_columns
                ])
                .rename({'labeltime_next': 'next_labeltime'})
            )

            final_target_data = pl.concat([has_next_state, no_next_state])

            (
                final_target_data
                # Again, sort out the "nulls" for each in way that is consistent with the dtypes
                .with_columns([
                    pl.when(pl.col('next_' + col).is_null())
                    .then(pl.lit([-1 for _ in range(context_length)]).cast(pl.List(pl.Int16)).alias('next_' + col))
                    .otherwise('next_' + col) for col in input_feature_columns
                    if 'encoded_feature' in col or 'time' in col  # this is inclusive of feature/featuretime/delta_time
                ])

                .with_columns([
                    pl.when(pl.col('next_' + col).is_null())
                    .then(pl.lit([np.nan for _ in range(context_length)]).alias('next_' + col))
                    .otherwise('next_' + col) for col in input_feature_columns if 'value' in col
                ])
                # Collect and save the data
                .collect()
                .write_parquet(os.path.join(dataframe_path, f'dataframe{idx:03}.parquet'))
            )


def create_hdf5_array(target_path, batch_dir, features, label_features, context_window):
    # Create the .hdf5 file and some blank placeholders (for use later)
    is_train = 'train' in target_path
    h5_array = h5py.File(target_path, 'w')
    h5_array = create_hdf5_datasets(h5_array, features, label_features, context_window, is_train=is_train)
    next_size = next_real_size = 0

    batch_dir_files = sorted(os.listdir(batch_dir))
    scaling_path = './data/scaling_data.parquet'

    # Iterate through each batch of data (i.e., each .parquet file)
    for batch_file_idx, batch_file in progress_bar(batch_dir_files, with_time=True):
        # Load the .parquet dataframe
        dataframe_batch = pl.scan_parquet(os.path.join(batch_dir, batch_file))

        # Convert the dataframe to numpy arrays
        (feature_vals, timepoint_vals, value_vals, delta_time_vals, delta_value_vals,
         label_vals, next_feature_vals, next_timepoint_vals, next_value_vals,
         next_delta_time_vals, next_delta_value_vals) = create_numpy_arrays_from_dataframe(dataframe_batch,
                                                                                           context_window,
                                                                                           label_features)

        # Resize the h5_array datasets to accommodate the new data
        current_size = next_size
        next_size = current_size + value_vals.shape[0]

        h5_array = resize_hdf5_datasets(h5_array, next_size, label_features, context_window)

        # Compress and write data to the datasets
        for embedding, (data, next_data), dtype in [
            ['features', (feature_vals, next_feature_vals), np.int16],
            ['timepoints', (timepoint_vals, next_timepoint_vals), np.int16],
            ['values', (value_vals, next_value_vals), np.float32],
            ['delta_time', (delta_time_vals, next_delta_time_vals), np.int16],
            ['delta_value', (delta_value_vals, next_delta_value_vals), np.float32]
        ]:
            h5_array[embedding][current_size:] = data.astype(dtype=dtype)[:, :context_window]
            h5_array[f'next_{embedding}'][current_size:] = next_data.astype(dtype=dtype)[:, :context_window]

        h5_array['labels'][current_size:] = label_vals

        if not is_train:
            continue

        # Update the scaling data

        # We want to find min, max, mean, and std values for our (relative) timepoints (using sse to update std)
        # For timepoints, we can use all the data at once (all features share a similar scale)

        # Flatten the data and remove all the nans
        real_mask = np.where(feature_vals.flatten() > -1)[0]
        timepoint_vals = timepoint_vals.flatten()[real_mask]

        # Update the scaling values for timepoints and delta_time, as the simpler step
        current_min = h5_array['min']['timepoints'][:]
        current_max = h5_array['max']['timepoints'][:]
        current_mean = h5_array['mean']['timepoints'][:]
        current_sse = h5_array['sse']['timepoints'][:]

        next_real_size += real_mask.size

        # Now we can calculate the min, max, mean, and std
        min_vals = np.nanmin(np.concatenate((timepoint_vals, current_min), 0), 0).reshape(1)
        max_vals = np.nanmax(np.concatenate((timepoint_vals, current_max), 0), 0).reshape(1)

        h5_array['min']['timepoints'][:] = min_vals
        h5_array['max']['timepoints'][:] = max_vals

        # Calculate the mean and std for each feature - we will use a special formula for continual updates
        # Calculate our update terms
        update_term = timepoint_vals - current_mean

        update_mean_term = np.nansum(update_term / np.maximum(next_real_size, 1), 0)

        update_mean = current_mean + update_mean_term

        update_sse_term = np.nansum(update_term * (timepoint_vals - update_mean), 0)

        update_sse = current_sse + update_sse_term

        update_std = np.sqrt(update_sse / np.maximum(next_real_size, 1))

        h5_array['mean']['timepoints'][:] = update_mean
        h5_array['std']['timepoints'][:] = update_std
        h5_array['sse']['timepoints'][:] = update_sse

    # Update the scaling for all other parameters (using our pre-prepared scaling.parquet file)
    if not is_train:
        h5_array.close()
        return

    # Delete the sse dataset, as this is no longer required
    del h5_array['sse']

    scaling_df = pl.read_parquet(scaling_path)
    for idx, feature in enumerate(features):
        current_feature_df = scaling_df.filter(pl.col('str_feature') == feature)
        if current_feature_df.is_empty():
            continue
        # Make sure our idx/feature mapping is intact
        assert current_feature_df.select('encoded_feature').item() == idx
        for group in ['mean', 'std', 'max', 'min']:
            for embedding in ['values', 'delta_value', 'delta_time']:
                h5_array[group][embedding][feature][:] = (
                    current_feature_df
                    .select(f'{embedding}_{group}')
                    .to_series().item()
                )

    # Close the newly created h5 array
    h5_array.close()
    return


def create_hdf5_datasets(h5_array: h5py.File, features: list = None, label_features: list = None,
                         context_window: int = 400, is_train: bool = True):
    empty_data_shape = (0, context_window, 1)
    max_data_shape = (10 ** 8, context_window, 1)

    empty_label_shape = (0, len(label_features))
    max_label_shape = (10 ** 8, len(label_features))

    # Create a dataset for encoded features for each token tuple
    # Use int16 when possible to save space (we will use -1 instead of nan)
    for embedding, dtype in [['features', np.int16], ['timepoints', np.int16], ['values', np.float32],
                             ['delta_time', np.int16], ['delta_value', np.float32]]:
        for i in ['', 'next_']:
            h5_array.create_dataset(name=i + embedding,
                                    shape=empty_data_shape,
                                    maxshape=max_data_shape,
                                    compression='gzip',
                                    compression_opts=9,
                                    dtype=dtype)

    # Create a dataset for the labels
    h5_array.create_dataset(name='labels',
                            shape=empty_label_shape,
                            maxshape=max_label_shape,
                            compression='gzip',
                            compression_opts=9,
                            dtype=np.float32)

    # Create datasets for standardisation/normalisation values
    if is_train:
        for scale in ['min', 'max', 'mean', 'std', 'sse']:
            h5_array.create_group(scale)

            if scale in ['mean', 'std', 'sse']:
                h5_array[scale].create_dataset(name='timepoints',
                                               data=np.zeros(1),
                                               dtype=np.float32)
            else:
                h5_array[scale].create_dataset(name='timepoints',
                                               # We want these values overwritten immediately,
                                               # so max is set to -inf and min is set +inf
                                               data=np.inf * np.ones(1) * -1 if scale == 'max' else np.inf * np.ones(1),
                                               dtype=np.float32)

            for embedding in ['values', 'delta_value', 'delta_time']:
                h5_array[scale].create_group(embedding)

                for feature in features:
                    if scale in ['mean', 'std', 'sse']:
                        h5_array[scale][embedding].create_dataset(name=feature,
                                                                  data=np.zeros(1),
                                                                  dtype=np.float32)
                    else:
                        h5_array[scale][embedding].create_dataset(name=feature,
                                                                  # We want these values overwritten immediately,
                                                                  # so max is set to -inf and min is set +inf
                                                                  data=np.inf * np.ones(
                                                                      1) * -1 if scale == 'max' else np.inf * np.ones(1),
                                                                  dtype=np.float32)

    # Lastly, store the actual feature names and label names
    h5_array.create_dataset(name='feature_names',
                            data=features)
    h5_array.create_dataset(name='label_names',
                            data=label_features)

    return h5_array


def create_numpy_arrays_from_dataframe(df: pl.LazyFrame, context_window: int = 2000, label_features: list = None):
    f = lambda x, column: x.select(column).explode(column).collect().to_numpy().reshape(-1, context_window, 1)
    feature_vals = f(df, 'encoded_feature')
    timepoint_vals = f(df, 'featuretime')
    value_vals = f(df, 'value')
    delta_time_vals = f(df, 'delta_time')
    delta_value_vals = f(df, 'delta_value')
    next_feature_vals = f(df, 'next_encoded_feature')
    next_timepoint_vals = f(df, 'next_featuretime')
    next_value_vals = f(df, 'next_value')
    next_delta_time_vals = f(df, 'next_delta_time')
    next_delta_value_vals = f(df, 'next_delta_value')

    label_vals = df.select(label_features).collect().to_numpy()

    return (feature_vals, timepoint_vals, value_vals, delta_time_vals, delta_value_vals, label_vals,
            next_feature_vals, next_timepoint_vals, next_value_vals, next_delta_time_vals, next_delta_value_vals)


def create_labels_for_mimic(combined_data, admissions, patients, input_window_size, state_delay, next_state_window=24):
    # Create our labels DataFrame - label is every new measurement starttime (the "state marker")
    labels = (
        combined_data
        .select(['subject_id', 'starttime', 'patientweight'])
        .unique()
        .sort(by=['subject_id', 'starttime'])
        .lazy()
    )

    # Get the death labels
    labels = get_death_labels_for_mimic(labels, admissions, patients)
    label_columns = ['subject_id', 'starttime', 'patientweight', '1-day-died', '3-day-died', '7-day-died',
                     '14-day-died', '28-day-died']

    # Create our start/end inclusion times for input data (and save the DataFrame for the next step)
    (
        labels
        .sort(by=['subject_id', 'starttime'])
        .select(label_columns +
                [(pl.col('starttime') - pl.duration(hours=input_window_size)).alias('start_inclusion'),
                 pl.col('starttime').alias('end_inclusion')])
        .collect()
        .write_parquet('./data/mimic/parquet/labels.parquet')
    )

    # Get our next_state times (if they exist)
    labels = pl.scan_parquet('./data/mimic/parquet/labels.parquet')
    patient_ids = labels.select(pl.col('subject_id').unique()).collect().to_series().to_list()
    labels = get_next_label_state(labels, 'mimic', patient_ids, state_delay, next_state_window)

    # Add age and gender data
    # To get age, you must do the following:
    # - Get the anchor_age and anchor_year from `patients`
    # - Calculate 'starttime' - 'anchor_year'
    # - Add this result to anchor_age
    # To get gender, you must do the following:
    # - Get the 'gender' column from patients.csv, and convert 'F'/'M' to binary values
    labels = (
        labels
        .join(patients.lazy()
              .select(['subject_id', 'gender', 'anchor_age', 'anchor_year']), on='subject_id', how='inner')
        .with_columns([
            # Turn gender into a binary value
            pl.when(pl.col('gender') == "F")
            .then(pl.lit(1).cast(pl.UInt8).alias('gender'))
            .otherwise(pl.lit(0)),

            # Calculate age
            (pl.col('starttime').dt.year() - pl.col('anchor_year') + pl.col('anchor_age')).cast(pl.UInt8).alias('age')
        ])
        .drop(['anchor_age', 'anchor_year'])
    )

    # Set the column order
    label_col_order = ['subject_id', 'labeltime', 'labeltime_next', 'start_inclusion',
                       'end_inclusion', '1-day-died', '3-day-died', '7-day-died', '14-day-died',
                       '28-day-died', 'age', 'gender', 'patientweight']

    # Collect the labels
    labels = (
        labels
        .rename({'starttime': 'labeltime', 'starttime_next': 'labeltime_next'})
        .select(label_col_order).unique()
        # Fill the blank patient weights - start by filling with the previous value,
        # and insert a default based on gender for the rest.
        # Taken from train_ids: mean: F = 74, M = 86
        .with_columns([pl.col('patientweight').fill_null(strategy='forward').over('subject_id')])
        .with_columns([pl.when(pl.col('patientweight').is_null())
                      .then(pl.when(pl.col('gender') == 0)
                            .then(pl.lit(86.).alias('patientweight'))
                            .otherwise(pl.lit(74.).alias('patientweight')))
                      .otherwise(pl.col('patientweight'))])
        # We've accumulated a few LazyFrame operations by this point, so time to collect.
        .collect()
    )

    return labels


def create_labels_for_sicdb(combined_data, input_window_size, state_delay, next_state_window=24):
    label_columns = ['subject_id', 'CaseID', 'starttime', 'patientweight', 'gender', 'age', '1-day-died',
                     '3-day-died', '7-day-died', '14-day-died', '28-day-died']

    # Create our labels DataFrame - label is every new measurement starttime (the "state marker")
    labels = (
        combined_data
        .select(label_columns)
        .unique()
    )

    # Create our start/end inclusion times for input data (and save the DataFrame for the next step)
    labels = (
        labels
        .sort(by=['subject_id', 'CaseID', 'starttime'])
        .with_columns([(pl.col('starttime') - pl.duration(hours=input_window_size)).alias('start_inclusion'),
                       pl.col('starttime').alias('end_inclusion')])
    )

    # Get out next_state times (if they exist)
    #  - For cross-compatibility with mimic, we need to separate out a couple of columns before applying these functions
    labels = reverse_ids_for_sicdb(labels)
    label_excl = labels.select('subject_id_original', 'subject_id', 'starttime', 'gender', 'age')
    labels = labels.drop('gender', 'age', 'subject_id_original')

    labels.write_parquet('./data/sicdb/parquet/labels.parquet')
    labels = pl.scan_parquet('./data/sicdb/parquet/labels.parquet')

    case_ids = labels.select(pl.col('subject_id').unique()).collect().to_series().to_list()
    labels = get_next_label_state(labels, 'sicdb', case_ids, state_delay, next_state_window)
    labels = labels.join(label_excl.lazy(), on=['subject_id', 'starttime'], how='inner')
    labels = reverse_ids_for_sicdb(labels)

    # Set the column order
    label_col_order = ['subject_id', 'CaseID', 'labeltime', 'labeltime_next', 'start_inclusion',
                       'end_inclusion', '1-day-died', '3-day-died', '7-day-died',
                       '14-day-died', '28-day-died', 'age', 'gender', 'patientweight']

    # Collect the labels
    labels = (
        labels
        .rename({'starttime': 'labeltime', 'starttime_next': 'labeltime_next'})
        .select(label_col_order).unique()
        # We've accumulated a few LazyFrame operations by this point, so time to collect.
        .collect()
    )

    return labels


def delete_parquet_files(data_option: str):
    if data_option == 'mimic':
        for group in ['train', 'val', 'test']:
            shutil.rmtree(f'./data/mimic/{group}/dataframe_{group}', ignore_errors=True)
        for file in ['combined_data', 'encoded_input_data', 'labels']:
            os.remove(f'./data/mimic/parquet/{file}.parquet')
    elif data_option == 'sicdb':
        for group in ['train', 'val', 'test']:
            shutil.rmtree(f'./data/sicdb/{group}/dataframe_{group}', ignore_errors=True)
        for file in ['combined_data', 'encoded_input_data', 'labels']:
            os.remove(f'./data/sicdb/parquet/{file}.parquet')
    else:
        raise ValueError(f'Invalid data option: {data_option}')
    return


def encode_combined_data_for_mimic(combined_data):
    """
    Create the encodings for our features
    """
    # First, create separate drug variables for rate and bolus, and create a single 'value' column
    encoded_input_data = split_labels_to_rate_and_bolus(combined_data)
    encoded_input_data = encoded_input_data.rename({'starttime': 'featuretime'})

    # Convert our feature names to integer encodings
    features = encoded_input_data.select(pl.col('feature').cast(pl.Utf8)).unique().to_series().sort().to_list()
    features.append('age')
    features.append('gender')
    features.append('patientweight')

    feature_encoding = (
        pl.DataFrame({'str_feature': features,
                      'encoded_feature': np.array([i for i in range(len(features))], dtype=np.int16)})
        .with_columns([pl.col('str_feature').cast(pl.Categorical('lexical'))])
    )

    # Encode the features in combined_data
    encoded_input_data = (
        encoded_input_data
        .rename({'feature': 'str_feature'})
        .join(feature_encoding, on='str_feature', how='inner')
    )

    # Keep track of the encodings for all our features (by category), for use later on
    age_encoding = np.int16(feature_encoding.filter(pl.col('str_feature') == 'age').select('encoded_feature').item())
    gender_encoding = np.int16(
        feature_encoding.filter(pl.col('str_feature') == 'gender').select('encoded_feature').item())
    weight_encoding = np.int16(
        feature_encoding.filter(pl.col('str_feature') == 'patientweight').select('encoded_feature').item())

    str_features, all_encoded_features = (encoded_input_data.select('str_feature', 'encoded_feature')
                                          .unique().sort('str_feature'))
    drug_names_encoded, lab_names_encoded = [], []
    for str_feature, encoded_feature in zip(str_features, all_encoded_features):
        if 'rate' in str_feature:
            drug_names_encoded.extend([encoded_feature])
        elif 'bolus' in str_feature:
            continue
        else:
            lab_names_encoded.extend([encoded_feature])

    encodings = {'age': age_encoding, 'gender': gender_encoding, 'weight': weight_encoding,
                 'drug_names': drug_names_encoded, 'lab_names': lab_names_encoded, 'all_features': all_encoded_features}

    return encoded_input_data, features, feature_encoding, encodings


def encode_combined_data_for_sicdb(combined_data):
    """
    Create the encodings for our features
    """
    # First, create separate drug variables for rate and bolus, and create a single 'value' column
    encoded_input_data = split_labels_to_rate_and_bolus(combined_data, 'CaseID')
    encoded_input_data = encoded_input_data.rename({'starttime': 'featuretime'})

    # Import the MIMIC feature_encoding template
    feature_encoding = pl.read_parquet('./data/feature_encoding.parquet')

    # Encode the features in combined_data
    encoded_input_data = (
        encoded_input_data
        .rename({'feature': 'str_feature'})
        .join(feature_encoding, on='str_feature', how='inner')
    )

    # Keep track of the encodings for all our features (by category), for use later on
    age_encoding = np.int16(feature_encoding.filter(pl.col('str_feature') == 'age').select('encoded_feature').item())
    gender_encoding = np.int16(
        feature_encoding.filter(pl.col('str_feature') == 'gender').select('encoded_feature').item())
    weight_encoding = np.int16(
        feature_encoding.filter(pl.col('str_feature') == 'patientweight').select('encoded_feature').item())

    str_features, all_encoded_features = (encoded_input_data.select('str_feature', 'encoded_feature')
                                          .unique().sort('str_feature'))
    drug_names_encoded, lab_names_encoded = [], []
    for str_feature, encoded_feature in zip(str_features, all_encoded_features):
        if 'rate' in str_feature:
            drug_names_encoded.extend([encoded_feature])
        else:
            lab_names_encoded.extend([encoded_feature])

    encodings = {'age': age_encoding, 'gender': gender_encoding, 'weight': weight_encoding,
                 'drug_names': drug_names_encoded, 'lab_names': lab_names_encoded, 'all_features': all_encoded_features}

    return encoded_input_data, encodings


def get_antibiotic_names():
    return ['Aciclovir', 'Ambisome', 'Amikacin', 'Ampicillin', 'Ampicillin-Sulbactam', 'Azithromycin', 'Aztreonam',
            'Co-trimoxazole', 'Caspofungin', 'Cefazolin', 'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Ciprofloxacin',
            'Chloramphenicol', 'Clindamycin', 'Colistin', 'Daptomycin', 'Doxycycline', 'Ertapenem', 'Erythromycin',
            'Gentamicin', 'Levofloxacin', 'Meropenem', 'Linezolid', 'Micafungin', 'Metronidazole', 'Nafcillin',
            'Oxacillin', 'Piperacillin', 'Piperacillin-Tazobactam', 'Rifampin', 'Tigecycline', 'Tobramycin',
            'Vancomycin', 'Voriconazole', 'IVIG', 'Mannitol']


def get_cleaning_functions(data_option: str):
    return {'mimic': {'convert_offset_to_datetime': lambda x: x,
                      'change_lab_units': change_lab_units_for_mimic,
                      'change_antibiotic_doses': change_antibiotic_doses_for_mimic,
                      'change_drug_units': change_drug_units_for_mimic,
                      'change_basal_bolus': separate_basal_bolus_in_mimic,
                      'merge_overlapping_rates': merge_overlapping_rates,
                      'add_zero_rates': add_zero_rates,
                      'add_demographics': lambda x, y: x,
                      'remove_outliers': lambda x, ids: remove_outliers_from_mimic(x, ids)},

            'sicdb': {'convert_offset_to_datetime': convert_sicdb_offset_to_datetime,
                      'change_lab_units': change_lab_units_for_sicdb,
                      'change_antibiotic_doses': change_antibiotic_doses_for_sicdb,
                      'change_drug_units': change_drug_units_for_sicdb,
                      'change_basal_bolus': separate_basal_bolus_in_sicdb,
                      'merge_overlapping_rates': lambda x: merge_overlapping_rates(reverse_ids_for_sicdb(x)),
                      'add_zero_rates': lambda x: reverse_ids_for_sicdb(add_zero_rates(x)),
                      'add_demographics': add_demographics_for_sicdb,
                      'remove_outliers': lambda x, ids: remove_outliers_from_sicdb(x)}}[data_option]


def get_death_labels_for_mimic(df, admissions, patients):
    """
    Get the labels for death events.

    Admissions contains the deathtime for patients who died in hospital.
    Patients contains the date of death for patients who died after discharge.
    """
    # Refine admission DataFrame
    admissions = (admissions
                  .select(['subject_id', 'deathtime'])
                  .filter((pl.col('deathtime').is_null().all().over('subject_id'))
                          | (pl.col('deathtime').is_not_null()))
                  .unique())

    # Add in death times
    df = (
        df
        # Join with the `admissions` DataFrame, which contains admission deaths
        .join(admissions, on=['subject_id'], how='inner')
        # Join with the `patients` DataFrame (for deaths just after discharge)
        .join(patients.select(['subject_id', 'dod']).unique(), on='subject_id', how='inner')
        # When 'deathtime' is null but 'dod' occurs on the same day as some labels, use the final label as deathtime
        .with_columns([
            pl.when(pl.col('deathtime').is_null()
                    & pl.col('dod').is_not_null()
                    & (pl.col('starttime') >= pl.col('dod')).any().over('subject_id'))
            .then(pl.col('starttime').last().over('subject_id').alias('deathtime'))
            .otherwise(pl.col('deathtime'))
        ])
        # Now combine dod and deathtime into a single column
        .select(['subject_id', 'starttime', 'patientweight',
                 pl.when(pl.col('deathtime').is_not_null())
                .then(pl.col('deathtime'))
                .otherwise(pl.col('dod'))])
        # If 'deathtime' appears 2+ times, choose the later time
        .sort(by=['subject_id', 'starttime', 'deathtime'])
        .filter((pl.col('deathtime').is_null())
                | (pl.col('deathtime') == pl.col('deathtime').last().over('subject_id')))
    )

    # Convert 1-day, 3-day, 7-day, 14-day, and 28-day mortality into reward labels
    df = (
        df
        .with_columns([
            pl.when(pl.col('deathtime').is_not_null()
                    & (pl.col('deathtime') <= pl.col('starttime') + pl.duration(days=day)))
            # Predict death as a probability of 1
            .then(pl.lit(1).cast(pl.UInt8).alias(f'{day}-day-died'))
            # Represent survival as a probability of 0
            .otherwise(pl.lit(0).alias(f'{day}-day-died')) for day in [1, 3, 7, 14, 28]
        ]).unique())

    return df


def get_drug_names():
    return ['Amiodarone', 'Amiodarone', 'Amiodarone', 'Amiodarone', 'Aciclovir', 'Ambisome', 'Amikacin', 'Ampicillin',
            'Ampicillin-Sulbactam', 'Azithromycin', 'Aztreonam', 'Co-trimoxazole', 'Caspofungin', 'Cefazolin',
            'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Ciprofloxacin', 'Chloramphenicol', 'Clindamycin', 'Colistin',
            'Daptomycin', 'Doxycycline', 'Ertapenem', 'Erythromycin', 'Gentamicin', 'Levofloxacin', 'Meropenem',
            'Linezolid', 'Micafungin', 'Metronidazole', 'Nafcillin', 'Oxacillin', 'Piperacillin',
            'Piperacillin-Tazobactam', 'Rifampin', 'Tigecycline', 'Tobramycin', 'Vancomycin', 'Voriconazole',
            'Unfractionated Heparin', 'Levetiracetam', 'Phenytoin', 'Nitroglycerin', 'Nitroprusside', 'Labetalol',
            'FFP', 'Human Albumin Solution 25%', 'IVIG', 'Platelet infusion', 'PRBC', 'Furosemide', 'Furosemide',
            'Bumetanide', 'Regular Insulin', 'Hypertonic Saline', 'Mannitol', 'Aminophylline',
            'Sodium Bicarbonate 8.4%', 'Sodium Bicarbonate 8.4%', 'Fentanyl', 'Fentanyl', 'Fentanyl', 'Morphine',
            'Cisatracurium', 'Rocuronium', 'Vecuronium', 'Dexmedetomidine', 'Ketamine', 'Ketamine', 'Lorazepam',
            'Midazolam', 'Propofol', 'Propofol', 'Alteplase', 'Adrenaline', 'Dobutamine', 'Dopamine', 'Milrinone',
            'Noradrenaline', 'Vasopressin']


def get_lab_names():
    return ['ALT', 'AST', 'Albumin', 'ALP', 'Amylase', 'Anion Gap', 'Base Excess', 'Blood Gas pCO2', 'Blood Gas SpO2',
            'Blood Gas pO2', 'Urea', 'Ionised Calcium', 'Calcium', 'CRP', 'Chloride', 'Chloride', 'Creatinine',
            'Glucose', 'Bedside Glucose', 'Bedside Glucose', 'HCO3', 'Haematocrit', 'Haemoglobin', 'Lactate', 'LDH',
            'Lipase', 'pH', 'pH', 'Platelets', 'Potassium', 'Potassium', 'Prothrombin Time', 'Sodium', 'Sodium',
            'Bilirubin', 'Troponin - T', 'Blood Gas pCO2', 'WBC', #'Blood Gas pO2',

            # Temporary for SOFA!
            'FiO2', 'GCS - Eye', 'GCS - Motor', 'GCS - Verbal', 'MAP', 'On ventilation', 'Blood Gas PaO2',
            'Blood Gas PvO2'
    ]


def get_next_label_state(labels, data_group: str, patient_ids, next_state_start, next_state_window=24):
    """
    Get the next label state for each patient. Can result in OOM errors, so done lazily and in chunks.
    """
    chunk_size = 1000  # <- number of patient ids per chunk
    chunk_size = len(patient_ids) // chunk_size + 1
    os.makedirs(f'./data/{data_group}/labels_temp_dir', exist_ok=True)

    print('Finding the "next state" marker...')
    for idx, ids_chunk in progress_bar(np.array_split(patient_ids, chunk_size)):
        (
            labels
            .filter(pl.col('subject_id').is_in(ids_chunk))
            .join(labels.select(['subject_id', 'starttime']),
                  on=['subject_id'], how='inner', suffix='_next')
            .filter(pl.col('starttime_next').is_between(pl.col('starttime') + pl.duration(hours=next_state_start),
                                                        pl.col('starttime') + pl.duration(hours=next_state_start
                                                                                                + next_state_window))
                    | (pl.col('starttime_next') == pl.col('starttime')))

            .with_columns([
                pl.when(pl.col('starttime_next') == pl.col('starttime'))
                .then(pl.datetime(9999, 1, 1).alias('starttime_next'))
                .otherwise(pl.col('starttime_next'))
            ])
            .sort(by=['subject_id', 'starttime', 'starttime_next'])
            .group_by(['subject_id', 'starttime', 'patientweight', '1-day-died', '3-day-died',
                       '7-day-died', '14-day-died', '28-day-died',
                       'start_inclusion', 'end_inclusion'])
            .agg(pl.col('starttime_next').first())
            .with_columns([pl.col('starttime_next').replace(pl.datetime(9999, 1, 1), None)])
            .collect()
            .write_parquet(f'./data/{data_group}/labels_temp_dir/labels_temp_{idx}.parquet')
        )

    (
        pl.scan_parquet(f'./data/{data_group}/labels_temp_dir/*.parquet')
        .sink_parquet(f'./data/{data_group}/parquet/labels.parquet')
    )

    labels = pl.scan_parquet(f'./data/{data_group}/parquet/labels.parquet')
    shutil.rmtree(f'./data/{data_group}/labels_temp_dir')

    return labels


def get_patient_ids(data_option: str, combined_data):
    # Define our patient_ids
    patient_ids = combined_data.select('subject_id').unique().to_series().to_list()
    np.random.seed(42)
    np.random.shuffle(patient_ids)

    if data_option == 'mimic':
        train_idx = round(0.8 * len(patient_ids))  # Set train proportion to 80%
        val_idx = round(0.9 * len(patient_ids))  # Set val (and test) proportion to 10%

        train_patient_ids, val_patient_ids, test_patient_ids = np.split(
            patient_ids, [train_idx, val_idx]
        )
    elif data_option == 'sicdb':
        train_patient_ids, val_patient_ids, test_patient_ids = np.array([]), np.array([]), patient_ids

    else:
        raise ValueError(f'Invalid data option: {data_option}')

    return train_patient_ids, val_patient_ids, test_patient_ids


def get_scaling_data_for_mimic(encoded_input_data, labels, train_patient_ids, input_window_size,
                               age_encoding, gender_encoding, weight_encoding):
    def create_feature_stats(df, feature_name, encoded_feature_number):
        return (
            df
            .select(
                [pl.lit(feature_name).cast(pl.Categorical('lexical')).alias('str_feature'),
                 pl.lit(encoded_feature_number).cast(pl.Int16).alias('encoded_feature'),
                 pl.col(feature_name).max().alias('values_max'),
                 pl.col(feature_name).min().alias('values_min'),
                 pl.col(feature_name).mean().alias('values_mean'),
                 pl.col(feature_name).std().alias('values_std')] +
                [pl.lit(val).alias(col) for val, col in fill_null_scales]
            )
        )

    train_labels = labels.filter(pl.col('subject_id').is_in(train_patient_ids))
    age_stats = train_labels.select(pl.col('age').cast(pl.Float64))
    gender_stats = train_labels.select(pl.col('gender').cast(pl.Float64))
    weight_stats = train_labels.select(pl.col('patientweight').cast(pl.Float64))

    scaling_data = (
        encoded_input_data
        # Very important that we only calculate for the train patient ids to avoid data leakage
        .filter(pl.col('subject_id').is_in(train_patient_ids))
        .sort('subject_id', 'featuretime')
        .with_columns([
            (pl.col('value') - pl.col('value').shift(1).over(['subject_id', 'str_feature'])).alias('delta_value'),
            (pl.col('featuretime') - pl.col('featuretime').shift(1).over(['subject_id', 'str_feature'])).alias(
                'delta_time')
        ])
        .with_columns([
            pl.when(pl.col('delta_time') > pl.duration(hours=input_window_size))
            .then(pl.lit(None).alias('delta_value'))
            .otherwise(pl.col('delta_value'))
        ])
        # Delta time and delta value done separately to avoid any interactions
        .with_columns([
            pl.when(pl.col('delta_time') > pl.duration(hours=input_window_size))
            .then(pl.lit(None).alias('delta_time'))
            .otherwise(pl.col('delta_time').dt.total_minutes())
        ])
        .group_by('str_feature', 'encoded_feature')
    )
    # Assemble our required columns
    aggregate_columns = []
    for col in ['value', 'delta_value', 'delta_time']:
        aggregate_columns.extend([
            pl.col(col).max().alias(f'{col}_max'),
            pl.col(col).min().alias(f'{col}_min'),
            pl.col(col).mean().alias(f'{col}_mean'),
            pl.col(col).std().alias(f'{col}_std')
        ])

    scaling_data = (
        scaling_data
        .agg(aggregate_columns)
        .select('str_feature', 'encoded_feature', pl.exclude('str_feature', 'encoded_feature'))
        .rename({f'value_{scale}': f'values_{scale}' for scale in ['max', 'min', 'mean', 'std']})
    )

    # Add in age/gender/weight
    fill_null_scales = []
    for group in ['delta_value', 'delta_time']:
        for val, scale in [[1, 'max'], [0, 'min'], [0, 'mean'], [1, 'std']]:
            fill_null_scales.extend([
                [val, f'{group}_{scale}']
            ])

    age_stats = create_feature_stats(age_stats, 'age', age_encoding)
    gender_stats = create_feature_stats(gender_stats, 'gender', gender_encoding)
    weight_stats = create_feature_stats(weight_stats, 'patientweight', weight_encoding)
    scaling_data = pl.concat([scaling_data, pl.concat([age_stats, gender_stats, weight_stats])], how='vertical_relaxed')
    # Fill std nulls for binary/rare events etc
    scaling_data = scaling_data.with_columns([pl.col('values_std').fill_null(1.)]
                                             + [pl.col(col).fill_null(val) for val, col in fill_null_scales])

    return scaling_data


def get_variable_names_for_mimic():
    return {
        # Infusions
        # - Antiarrhythmics
        'Amiodarone': 'Amiodarone', 'Amiodarone 600/500': 'Amiodarone',
        'Amiodarone 450/250': 'Amiodarone', 'Amiodarone 150/100': 'Amiodarone',

        # - Antibiotics
        'Acyclovir': 'Aciclovir', 'Ambisome': 'Ambisome', 'Amikacin': 'Amikacin',
        'Ampicillin': 'Ampicillin', 'Ampicillin/Sulbactam (Unasyn)': 'Ampicillin-Sulbactam',
        'Azithromycin': 'Azithromycin', 'Aztreonam': 'Aztreonam', 'Bactrim (SMX/TMP)': 'Co-trimoxazole',
        'Caspofungin': 'Caspofungin', 'Cefazolin': 'Cefazolin', 'Cefepime': 'Cefepime', 'Ceftazidime': 'Ceftazidime',
        'Ceftriaxone': 'Ceftriaxone', 'Ciprofloxacin': 'Ciprofloxacin', 'Chloramphenicol': 'Chloramphenicol',
        'Clindamycin': 'Clindamycin', 'Colistin': 'Colistin',
        'Daptomycin': 'Daptomycin', 'Doxycycline': 'Doxycycline',
        'Ertapenem': 'Ertapenem', 'Erythromycin': 'Erythromycin',
        'Gentamicin': 'Gentamicin', 'Levofloxacin': 'Levofloxacin',
        'Meropenem': 'Meropenem', 'Linezolid': 'Linezolid', 'Micafungin': 'Micafungin',
        'Metronidazole': 'Metronidazole', 'Nafcillin': 'Nafcillin', 'Oxacillin': 'Oxacillin',
        'Piperacillin': 'Piperacillin',
        'Piperacillin/Tazobactam (Zosyn)': 'Piperacillin-Tazobactam',
        'Rifampin': 'Rifampin', 'Tigecycline': 'Tigecycline',
        'Tobramycin': 'Tobramycin',
        'Vancomycin': 'Vancomycin', 'Voriconazole': 'Voriconazole',

        # - Anticoagulants
        'Heparin Sodium': 'Unfractionated Heparin',

        # - Anticonvulsants
        'Levetiracetam (Keppra)': 'Levetiracetam', 'Fosphenytoin': 'Phenytoin',

        # - Antihypertensives
        'Nitroglycerin': 'Nitroglycerin', 'Nitroprusside': 'Nitroprusside',

        # - Beta blockers / CCBs
        'Labetalol': 'Labetalol',

        # - Blood products
        'Fresh Frozen Plasma': 'FFP', 'Albumin 25%': 'Human Albumin Solution 25%',
        'IV Immune Globulin (IVIG)': 'IVIG', 'Platelets': 'Platelet infusion', 'Packed Red Blood Cells': 'PRBC',

        # - Diuretics
        'Furosemide (Lasix)': 'Furosemide', 'Furosemide (Lasix) 250/50': 'Furosemide',
        'Bumetanide (Bumex)': 'Bumetanide',

        # - Glucose control
        'Insulin - Regular': 'Regular Insulin',

        # - Hypertonic saline / mannitol
        'NaCl 3% (Hypertonic Saline)': 'Hypertonic Saline', 'Mannitol': 'Mannitol',

        # - Miscellaneous
        'Aminophylline': 'Aminophylline', 'Sodium Bicarbonate 8.4%': 'Sodium Bicarbonate 8.4%',
        'Sodium Bicarbonate 8.4% (Amp)': 'Sodium Bicarbonate 8.4%',

        # - Opioids
        'Fentanyl': 'Fentanyl', 'Fentanyl (Concentrate)': 'Fentanyl', 'Fentanyl (Push)': 'Fentanyl',
        'Morphine Sulfate': 'Morphine',

        # - Paralytics
        'Cisatracurium': 'Cisatracurium', 'Rocuronium': 'Rocuronium', 'Vecuronium': 'Vecuronium',

        # - Sedatives incl benzos
        'Dexmedetomidine (Precedex)': 'Dexmedetomidine', 'Ketamine': 'Ketamine', 'Ketamine (Intubation)': 'Ketamine',
        'Lorazepam (Ativan)': 'Lorazepam', 'Midazolam (Versed)': 'Midazolam',
        'Propofol': 'Propofol', 'Propofol (Intubation)': 'Propofol',

        # - Thrombolytics
        'Alteplase (TPA)': 'Alteplase',

        # - Vasopressors / inotropes
        'Epinephrine': 'Adrenaline', 'Dobutamine': 'Dobutamine', 'Dopamine': 'Dopamine',
        'Milrinone': 'Milrinone', 'Norepinephrine': 'Noradrenaline', 'Vasopressin': 'Vasopressin',

        # Labs
        'ALT': 'ALT', 'AST': 'AST', 'Albumin': 'Albumin', 'Alkaline Phosphate': 'ALP', 'Amylase': 'Amylase',
        'Anion gap': 'Anion Gap', 'Arterial Base Excess': 'Base Excess', 'Arterial CO2 Pressure': 'Blood Gas pCO2',
        'Arterial O2 Saturation': 'Blood Gas SpO2',  # 'Arterial O2 pressure': 'Blood Gas pO2',
        'Arterial O2 pressure': 'Blood Gas PaO2',
        'BUN': 'Urea', 'Ionized Calcium': 'Ionised Calcium',
        'Calcium non-ionized': 'Calcium', 'C Reactive Protein (CRP)': 'CRP', 'Chloride (serum)': 'Chloride',
        'Chloride (whole blood)': 'Chloride', 'Creatinine (serum)': 'Creatinine', 'Glucose (serum)': 'Glucose',
        'Glucose (whole blood)': 'Bedside Glucose', 'Glucose finger stick (range 70-100)': 'Bedside Glucose',
        'HCO3 (serum)': 'HCO3', 'Hematocrit (serum)': 'Haematocrit', 'Hemoglobin': 'Haemoglobin',  # 'INR': 'INR',
        'Lactic Acid': 'Lactate', 'LDH': 'LDH', 'Lipase': 'Lipase', 'PH (Arterial)': 'pH', 'PH (Venous)': 'pH',
        'Platelet Count': 'Platelets', 'Potassium (serum)': 'Potassium', 'Potassium (whole blood)': 'Potassium',
        'Prothrombin time': 'Prothrombin Time', 'Sodium (serum)': 'Sodium', 'Sodium (whole blood)': 'Sodium',
        'Total Bilirubin': 'Bilirubin', 'Troponin-T': 'Troponin - T', 'Venous CO2 Pressure': 'Blood Gas pCO2',
        #'Venous O2 Pressure': 'Blood Gas pO2',
        'Venous O2 Pressure': 'Blood Gas PvO2',
        'WBC': 'WBC',

        # TEMPORARY FOR SOFA VALIDATION
        'Inspired O2 Fraction': 'FiO2', 'PEEP set': 'On ventilation',
        'GCS - Eye Opening': 'GCS - Eye', 'GCS - Motor Response': 'GCS - Motor',
        'GCS - Verbal Response': 'GCS - Verbal', 'Arterial Blood Pressure mean': 'MAP',
        'Non Invasive Blood Pressure mean': 'MAP',
    }


def get_variable_names_for_sicdb():
    return {
        # Infusions
        # - Antiarrhythmics
        'AMIOdaron': 'Amiodarone',

        # - Antibiotics
        'Acyclovir': 'Aciclovir', 'Liposomales Amphotericin B': 'Ambisome',
        'Amikacin': 'Amikacin', 'Ampicillin': 'Ampicillin', 'Ampicillin/Sulbactam': 'Ampicillin-Sulbactam',
        'Azithromycin': 'Azithromycin', 'Aztreonam': 'Aztreonam', 'Trimethoprim/Sulfametoxazol': 'Co-trimoxazole',
        'Sulfametoxazol+Trimethoprim': 'Co-trimoxazole', 'Caspofungin': 'Caspofungin', 'Cefazolin': 'Cefazolin',
        'Cefepim': 'Cefepime', 'Ceftazidim': 'Ceftazidime', 'Ceftriaxon': 'Ceftriaxone',
        'Ciprofloxacin': 'Ciprofloxacin', 'Clindamycin': 'Clindamycin', 'Colistin': 'Colistin',
        'Daptomycin': 'Daptomycin', 'Doxycyclin': 'Doxycycline', 'Doxycyclin monohydrat': 'Doxycycline',
        'Ertapenem': 'Ertapenem', 'Erythromycin': 'Erythromycin', 'Gentamycin': 'Gentamicin',
        'Levofloxacin': 'Levofloxacin', 'Meropenem': 'Meropenem', 'Linezolid': 'Linezolid',
        'Micafungin': 'Micafungin', 'Metronidazol': 'Metronidazole',
        'Piperacillin/Tazobactam': 'Piperacillin-Tazobactam', 'Rifampicin': 'Rifampin',
        'Tigecyclin': 'Tigecycline', 'Vancomycin': 'Vancomycin', 'Voriconazol': 'Voriconazole',

        # - Anticoagulants
        'Heparin': 'Unfractionated Heparin',

        # - Anticonvulsants
        'LevETIRAcetam': 'Levetiracetam', 'Phenytoin': 'Phenytoin',

        # - Antihypertensives
        'Nitroglycerin': 'Nitroglycerin', 'Natriumnitroprussid': 'Nitroprusside',

        # - Beta blockers / CCBs
        'Labetalol': 'Labetalol',

        # - Blood products
        'Octaplas': 'FFP', 'Humanalbumin 20%': 'Human Albumin Solution 20%',  # <- this will later have "20% removed
        'Humanalbumin 20% (Albunorm)': 'Human Albumin Solution 20%', 'Immunglobulin': 'IVIG',
        'Thrombozytenkonzentrat Einzelspender': 'Platelet infusion',
        'Thrombozytenkonzentrat gepoolt': 'Platelet infusion', 'Erythrozytenkonzentrat': 'PRBC',

        # - Diuretics
        'FUROsemid': 'Furosemide',  # NO bumetanide

        # - Glucose control
        'Insulin': 'Regular Insulin',

        # - Hypertonic saline / mannitol
        'Natriumchlorid3%': 'Hypertonic Saline', 'Mannitol (Mannit 10%)': 'Mannitol',
        'Mannitol (Mannit 20%)': 'Mannitol', 'Mannitol (Mannit 15%)': 'Mannitol',

        # - Miscellaneous
        'Natriumbikarbonat 8,4%': 'Sodium Bicarbonate 8.4%', 'Milrinon': 'Milrinone',

        # - Opioids
        'FentaNYL': 'Fentanyl', 'Morphin': 'Morphine',

        # - Paralytics
        'Cisatracurium': 'Cisatracurium', 'Cisatracurium besilat': 'Cisatracurium',
        'ROCuronium': 'Rocuronium',

        # - Sedatives incl benzos
        'Dexmedetomidin': 'Dexmedetomidine', 'KETAmin': 'Ketamine', 'LORazepam': 'Lorazepam',
        'Midazolam': 'Midazolam', 'Propofol 1%': 'Propofol', 'Propofol 2%': 'Propofol',

        # - Thrombolytics
        'Alteplase': 'Alteplase',

        # - Vasopressors / inotropes
        'EPINEPHrin': 'Adrenaline', 'DOBUtamin': 'Dobutamine', 'DOPamin': 'Dopamine',
        'Norepinephrin': 'Noradrenaline', 'Vasopressin': 'Vasopressin',

        # Labs
        'GPT ( ALT) (ZL)': 'ALT', 'GOT ( AST) (ZL)': 'AST', 'Serum-Albumin (ZL)': 'Albumin',
        'Alkalische Phosphatase (ZL)': 'ALP', 'Amylase (ZL)': 'Amylase', 'Anionen Gap (BGA)': 'Anion Gap',
        'BE (B) (BGA)': 'Base Excess', 'PCO2 (BGA)': 'Blood Gas pCO2',
        'SO2 (BGA)': 'Blood Gas SpO2', 'O2 SAT est (BGA)': 'Blood Gas SpO2',
        'PO2 (BGA)': 'Blood Gas pO2',
        'Harnstoff (ZL)': 'Urea', 'Calcium (i) (BGA)': 'Ionised Calcium',
        'Calcium (ZL)': 'Calcium', 'C-reaktives Protein (ZL)': 'CRP', 'Chlorid (ZL)': 'Chloride',
        'Chlorid (BGA)': 'Chloride', 'Kreatinin (ZL)': 'Creatinine', 'Glukose (ZL)': 'Glucose',
        'Glukose (BGA)': 'Bedside Glucose', 'Glucose (BG) (ZL)': 'Bedside Glucose',
        'HCO3 act (BGA)': 'HCO3', 'Hämatokrit (ZL)': 'Haematocrit', 'Hämoglobin (ZL)': 'Haemoglobin',
        'Lactat (BG) (ZL)': 'Lactate', 'Lactat (BGA)': 'Lactate', 'LDH (ZL)': 'LDH', 'Lipase (ZL)': 'Lipase',
        'PH (BGA)': 'pH', 'PH (BGA) gemV': 'pH', 'PH (BGA) kap': 'pH', 'PH (BGA) ven': 'pH',
        'Thrombocyten (ZL)': 'Platelets', 'Kalium (BG) (ZL)': 'Potassium', 'Kalium (BGA)': 'Potassium',
        'Kalium (ZL)': 'Potassium', 'Thrombinzeit (ZL)': 'Prothrombin Time',
        'Natrium (BG) (ZL)': 'Sodium', 'Natrium (ZL)': 'Sodium', 'Bilirubin (gesamt) (ZL)': 'Bilirubin',
        'hs Troponin T (ZL)': 'Troponin - T',  # <- different assay, same protein - needs unit conversions
        'Leukocyten (ZL)': 'WBC',
    }


def join_dfs(first_df, second_df):
    df = (
        first_df
        .join(second_df,
              on='itemid',
              how='inner')
        .with_columns([pl.col('label').cast(pl.Categorical('lexical'))])
        .drop('itemid')
    )

    return df


def load_files(directory, filename):
    filepath = os.path.join('./data', directory, filename + '.parquet')
    return pl.scan_parquet(filepath)


def load_mimic(variable_names: dict = None):
    admissions = load_files('mimic/parquet', 'admissions')
    chartevents = load_files('mimic/parquet', 'chartevents')
    d_items = load_files('mimic/parquet', 'd_items')
    inputevents = load_files('mimic/parquet', 'inputevents')
    patients = load_files('mimic/parquet', 'patients')

    # Filter and rename the d_items DataFrame according to the variable_names dictionary
    d_items = (
        d_items
        .filter(pl.col('label').is_in(variable_names.keys()))
        .select(['itemid', 'label'])
        .with_columns(
            pl.col('label').cast(pl.Utf8).replace(variable_names).cast(pl.Categorical('lexical'))
        )
    )

    # Select only the necessary columns from the chartevents and inputevents DataFrames
    chartevents = chartevents.select(
        ['subject_id', 'itemid', 'charttime', 'valuenum', pl.col('valueuom').cast(pl.Categorical('lexical'))]
    ).rename({'charttime': 'starttime'})

    inputevents = inputevents.select(
        ['subject_id', 'itemid', 'starttime', 'endtime', 'amount', pl.col('amountuom').cast(pl.Categorical('lexical')),
         'originalrate', 'rate', pl.col('rateuom').cast(pl.Categorical('lexical')), 'patientweight',
         pl.col('ordercategoryname').cast(pl.Categorical('lexical')),
         pl.col('ordercategorydescription').cast(pl.Categorical('lexical')),
         pl.col('statusdescription').cast(pl.Categorical('lexical')), 'orderid']
    )

    chartevents = join_dfs(chartevents, d_items)
    inputevents = join_dfs(inputevents, d_items)

    # Create our combined_data DataFrame
    with pl.StringCache():
        combined_data = (
            chartevents
            .join(inputevents, on=['subject_id', 'label', 'starttime'], how='full', coalesce=True)
            .select(
                ['subject_id', 'label', 'starttime', 'endtime', 'valuenum', 'valueuom', 'amount', 'amountuom',
                 'originalrate', 'rate', 'rateuom', 'ordercategoryname', 'ordercategorydescription', 'statusdescription',
                 'orderid', 'patientweight']
            )
            .collect()
            .unique()
            .sort(by=['subject_id', 'starttime', 'endtime', 'label'])
        )

    # Define our patient_ids
    train_patient_ids, val_patient_ids, test_patient_ids = get_patient_ids('mimic', combined_data)

    return admissions, combined_data, patients, (train_patient_ids, val_patient_ids, test_patient_ids)


def load_sicdb(variable_names: dict = None):
    cases = load_files('sicdb/parquet', 'cases')
    d_references = load_files('sicdb/parquet', 'd_references')
    laboratory = load_files('sicdb/parquet', 'laboratory')
    medication = load_files('sicdb/parquet', 'medication')

    # Change some column names for consistency
    cases = cases.rename({'PatientID': 'subject_id'})
    d_references = d_references.rename({'ReferenceValue': 'label', 'ReferenceGlobalID': 'itemid'})
    laboratory = laboratory.rename({'LaboratoryID': 'itemid', 'LaboratoryValue': 'value', 'Offset': 'starttime'})
    medication = medication.rename({'DrugID': 'itemid', 'PatientID': 'subject_id', 'Offset': 'starttime',
                                    'OffsetDrugEnd': 'endtime', 'id': 'orderid'})

    # Filter and rename the d_references DataFrame according to the variable_names dictionary
    d_references = (
        d_references
        .filter(pl.col('label').is_in(variable_names.keys()))
        .select(['itemid', 'label', 'ReferenceUnit'])
        .with_columns(
            pl.col('label').cast(pl.Utf8).replace(variable_names).cast(pl.Categorical('lexical'))
        )
    )

    # Select only the necessary columns from the medication and laboratory DataFrames
    medication = medication.select(
        ['CaseID', 'subject_id', 'itemid', 'starttime', 'endtime', 'IsSingleDose', 'Amount', 'AmountPerMinute',
         'orderid']
    )
    laboratory = (
        laboratory
        .select(['CaseID', 'itemid', 'starttime', 'value', 'LaboratoryType'])
        .join(cases.select('CaseID', 'subject_id'), on='CaseID', how='inner')
    )

    medication = join_dfs(medication, d_references)
    laboratory = join_dfs(laboratory, d_references)

    # Create our combined_data DataFrame
    combined_data = (
        medication
        .join(laboratory, on=['subject_id', 'CaseID', 'label', 'ReferenceUnit', 'starttime'], how='full', coalesce=True)
        .rename({'value': 'valuenum', 'ReferenceUnit': 'valueuom', 'Amount': 'amount', 'AmountPerMinute': 'rate'})
        .select(
            ['subject_id', 'CaseID', 'label', 'starttime', 'endtime', 'valuenum', 'valueuom', 'amount', 'rate',
             'IsSingleDose', 'orderid', 'LaboratoryType']
        )
        .collect()
        .unique()
        .sort(by=['subject_id', 'starttime', 'endtime', 'label'])
    )

    # Define our patient_ids
    train_patient_ids, val_patient_ids, test_patient_ids = get_patient_ids('sicdb', combined_data)

    return cases, combined_data, (train_patient_ids, val_patient_ids, test_patient_ids)


def merge_overlapping_rates(df):
    """
    There are occasions where two infusions of the same drug exist, with overlapping infusion periods.
    This function separates out and merges rates where necessary, to give two (dose equivalent) infusions that
    are temporally exclusive.

    This is done using the single_merge_iteration function, repeated 'n' times until no more overlapping rates
    exist.
    """
    has_rates = pl.col('rate').is_not_null()
    has_overlap_first_inf = pl.col('endtime') > pl.col('starttime').shift(-1).over(['subject_id', 'label'])
    has_overlap_second_inf = pl.col('starttime') < pl.col('endtime').shift(1).over(['subject_id', 'label'])
    has_overlap = has_overlap_first_inf | has_overlap_second_inf

    sort_by_start = lambda x: x.sort(by=['subject_id', 'label', 'starttime', 'endtime'])
    sort_by_end = lambda x: x.sort(by=['subject_id', 'label', 'endtime', 'starttime'])
    get_height = lambda x: x.filter(has_rates).filter(has_overlap).height

    sorted_by_start = get_height(sort_by_start(df))
    sorted_by_end = get_height(sort_by_end(df))

    print('Fixing overlapping infusions (be patient for SICDB!):')
    while sorted_by_start + sorted_by_end > 0:
        print(f'          Overlapping rows: {max(sorted_by_start, sorted_by_end)}')
        df = merge_overlapping_rates_single_iteration(df)
        sorted_by_start = get_height(sort_by_start(df))
        sorted_by_end = get_height(sort_by_end(df))

    print(f'          Overlapping rows: {max(get_height(sort_by_start(df)), get_height(sort_by_end(df)))}\n')
    return df


def merge_overlapping_rates_single_iteration(df):
    """
    Single iteration merge of overlapping rates - may leave some overlapping rates.
    """
    # Save column order
    column_order = df.columns
    df = df.with_columns([pl.col('orderid').cast(pl.Utf8).cast(pl.Categorical('lexical'))])

    # First, identify rates where this occurs (where an infusion starts before the previous infusion finished)
    has_rates = pl.col('rate').is_not_null()

    # Assuming sorted by endtime (then starttime), find overlapping rates
    has_overlap_first_inf = pl.col('endtime') > pl.col('starttime').shift(-1).over(['subject_id', 'label'])
    has_overlap_second_inf = pl.col('starttime') < pl.col('endtime').shift(1).over(['subject_id', 'label'])
    has_overlap = has_overlap_first_inf | has_overlap_second_inf

    # We have to check by overlaps using two different sorts, and then recombine
    overlap_df = (
        df
        .filter(has_rates)
        .sort(by=['subject_id', 'label', 'starttime', 'endtime'])
        .filter(has_overlap)
    )

    overlap_df = pl.concat([overlap_df, (df
                                         .filter(has_rates)
                                         # endtime and starttime switched around in this sort
                                         .sort(by=['subject_id', 'label', 'endtime', 'starttime'])
                                         .filter(has_overlap))
                            ]).unique().sort(by=['subject_id', 'label', 'starttime', 'endtime'])

    join_cols = ['subject_id', 'label', 'starttime', 'endtime', 'bolus', 'rate']
    if 'valuenum' in df.columns:
        join_cols.append('valuenum')

    remaining_df = (
        df
        .join(overlap_df, on=join_cols,
              how='anti', join_nulls=True)
    )

    # Explode out the rates into 1-minute intervals (smallest resolution possible)
    overlap_df = (
        overlap_df
        # Explode using starttime and endtime, into 1m intervals
        .with_columns(
            [pl.datetime_ranges(pl.col('starttime'), pl.col('endtime') - pl.duration(minutes=1),
                                '1m', eager=False).alias('starttime')])
        .explode('starttime')
        .with_columns([(pl.col('starttime') + pl.duration(minutes=1)).alias('endtime')])
    )

    # After exploding, aggregate rate by summing together
    first_new_segment = pl.col('orderid').shift(1).over(['subject_id', 'label']).is_null()
    next_new_segment = (pl.col('orderid') != pl.col('orderid').shift(1).over(['subject_id', 'label']))

    overlap_df = (
        overlap_df
        .sort(by=['subject_id', 'label', 'starttime', 'endtime', 'orderid'])
        .group_by(['subject_id', 'label', 'starttime', 'endtime'])
        .agg(pl.col('rate').sum(),
             pl.col('orderid').cast(pl.Utf8).str.concat('').cast(pl.Categorical('lexical')),
             # ^ This is used to give the overlapping minutes a new 'unique' id
             pl.exclude('rate', 'orderid').first())
        .sort(by=['subject_id', 'label', 'starttime', 'endtime'])
        # Create a marker for each new unique segment
        .with_columns([
            (first_new_segment | next_new_segment).alias('new_segment')
        ])
        .with_columns([
            pl.col('new_segment').cum_sum().over(['subject_id', 'label'])
        ])
    )

    # Collapse back again using the segment markers
    overlap_df = (
        overlap_df
        .sort(by=['subject_id', 'label', 'starttime', 'endtime'])
        .group_by(['subject_id', 'label', 'new_segment'])
        .agg(pl.col('starttime').first(),
             pl.col('endtime').last(),
             pl.exclude('starttime', 'endtime').first())
        # One final sort
        .sort(by=['subject_id', 'label', 'starttime', 'endtime'])
        .drop('new_segment')
    )

    # Concatenate the two dataframes back together
    df = (
        pl.concat([remaining_df.select(column_order), overlap_df.select(column_order)])
        .sort(by=['subject_id', 'starttime', 'endtime', 'label'])
    )

    return df


def progress_bar(iterable, with_time: bool = False):
    total_length = len(iterable)
    bar_length = 20
    bar = ' ' * bar_length
    sys.stdout.write('\r[>' + bar + '] ' + '0%')
    sys.stdout.flush()
    start = time.time() if with_time else None
    remaining = None
    for step, item in enumerate(iterable):
        yield step, item
        if with_time:
            end = time.time()
            duration = end - start  # in seconds
            speed = (step + 1) / duration  # in steps per second
            remaining = round((total_length - step + 1) / speed / 60, 1)  # in minutes

        progress = step / total_length
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length)
        if with_time:
            statement = '\r[' + bar + '] ' + str(int(progress * 100)) + '%, ETA: ' + str(remaining) + 'min'
        else:
            statement = '\r[' + bar + '] ' + str(int(progress * 100)) + '%'
        sys.stdout.write(statement.ljust(50))
        sys.stdout.flush()
    if total_length > 0:
        sys.stdout.write('\r[' + '=' * bar_length + '] 100%'.ljust(50) + '\n')
        sys.stdout.flush()


def remove_outliers_from_mimic(df, train_patient_ids):
    """
    Goal is to remove outliers in the data.

    To choose these, we have to balance having lots of data (avoid extreme quantiles) with having a good range
    of states (include extreme quantiles). We set drug quantiles to 0.005/0.995, and labs to 0.001/0.999.
    """

    def find_min_max(measurement, column, quantile):
        if quantile == 1:
            minimum, maximum = -np.inf, np.inf
        else:
            measurements = df.filter((pl.col('label') == column)
                                     & (pl.col('subject_id').is_in(train_patient_ids))).select(pl.col(measurement))
            minimum, maximum = measurements.quantile(1 - quantile).item(), measurements.quantile(quantile).item()

        return minimum, maximum

    # Create a dictionary that we can save later (for removing SICDB outliers)
    outlier_reference = {}

    # Start by tidying up patient weight - set to Null, we will forward fill these values later
    min_weight, max_weight = df.select([pl.col('patientweight').quantile(0.005).alias('min'),
                                        pl.col('patientweight').quantile(0.9995).alias('max')])
    df = df.with_columns([
        pl.when(pl.col('patientweight').is_between(min_weight, max_weight))
        .then(pl.col('patientweight'))
        .otherwise(pl.lit(None))
    ])

    outlier_reference['patientweight'] = {'min': min_weight, 'max': max_weight}

    # Get our variable names
    antibiotic_variables = get_antibiotic_names()
    drug_variables = get_drug_names()
    lab_variables = get_lab_names()

    # Iterate through all the variables in the DataFrame
    for variable in df.select('label').unique().to_series().to_list():
        # Antibiotics are set to binary boluses, so we can skip these
        if variable in antibiotic_variables:
            continue

        matches_variable = pl.col('label') == variable
        is_drug = variable in drug_variables
        is_lab = variable in lab_variables

        if not is_lab:
            if not is_drug:
                raise ValueError(f'Variable {variable} not found in known variables')
            # Unlikely to be relevant but filter out any 0 boluses
            else:
                df = (
                    df
                    .filter(~matches_variable |
                            (matches_variable & (
                                pl.col('rate').is_not_null() |
                                (pl.col('bolus') > 0)
                            )))
                )
        # rate_min set to 0 by default (as rate can always be stopped!)
        _, rate_max = find_min_max('rate', variable, 0.995) if is_drug else (0, np.inf)
        bolus_min, bolus_max = find_min_max('bolus', variable, 0.995) if is_drug else (0, np.inf)
        lab_min, lab_max = find_min_max('valuenum', variable, 0.999) if is_lab else (None, None)

        if is_lab:
            outlier_reference[variable] = {'min': lab_min, 'max': lab_max}
        else:
            outlier_reference[variable] = {'bolus': {'min': bolus_min, 'max': bolus_max},
                                           'rate': {'min': 0, 'max': rate_max}}

        bolus_not_in_range = ~pl.col('bolus').is_between(bolus_min, bolus_max)
        rate_not_in_range = ~pl.col('rate').is_between(0, rate_max)
        lab_not_in_range = ~pl.col('valuenum').is_between(lab_min, lab_max)

        df = (
            df.with_columns([
                # Clip drugs to range - this is an intervention and shouldn't be ignored completely
                pl.when(matches_variable & is_drug & rate_not_in_range)
                .then(pl.col('rate').clip(0, rate_max))
                .otherwise(pl.col('rate')),

                pl.when(matches_variable & is_drug & bolus_not_in_range)
                .then(pl.col('bolus').clip(bolus_min, bolus_max))
                .otherwise(pl.col('bolus')),

                # Set labs to null if outside range i.e., ignore these values completely
                pl.when(matches_variable & is_lab & lab_not_in_range)
                .then(pl.lit(None).alias('valuenum'))
                .otherwise(pl.col('valuenum')),

                pl.when(matches_variable & is_lab & lab_not_in_range)
                .then(pl.lit(None).alias('valueuom'))
                .otherwise(pl.col('valueuom'))
            ])
        )

    # Remove all the 'non-existent' rows
    df = (
        df.filter(
            pl.col('rate').is_not_null() | pl.col('valuenum').is_not_null() | pl.col('bolus').is_not_null())
    )

    # Save our outlier reference file (pickled)
    with open('./data/outlier_reference.pkl', 'wb') as f:
        pickle.dump(outlier_reference, f)
        f.close()

    return df


def remove_outliers_from_sicdb(df):
    """
    Remove outliers in the data, using the reference outliers generated from MIMIC.
    """
    # Load the saved outlier reference file
    with open('./data/outlier_reference.pkl', 'rb') as f:
        reference_outliers = pickle.load(f)
        f.close()

    # Get our variable names
    antibiotic_variables = get_antibiotic_names()
    drug_variables = get_drug_names()
    lab_variables = get_lab_names()

    # Then tidy up all other outliers
    for variable in reference_outliers.keys():
        # Tidy up patientweight as its own thing
        if variable == 'patientweight':
            # For SICDB, we are a bit limited by having patientweight per admission rather than per drug.
            # For the small number of subjects where we are outside the range (37.5 - 200), we will keep
            # the value as it is if not 0, and if it is 0, set to ? and ? (for F and M, respectively).
            is_female = pl.col('gender') == 1
            value_is_zero = pl.col('patientweight') == 0

            df = (
                df
                .with_columns([
                    pl.when(value_is_zero)
                    .then(pl.when(is_female)
                          # Mean weight for women = 74kg
                          .then(pl.lit(74.).alias('patientweight'))
                          # Mean weight for men = 86kg
                          .otherwise(pl.lit(86.).alias('patientweight')))
                    .otherwise(pl.col('patientweight'))
                ])
            )

            continue

        elif variable in antibiotic_variables:
            continue

        matches_variable = pl.col('label') == variable
        is_drug = variable in drug_variables
        is_lab = variable in lab_variables

        if is_lab:
            in_range = pl.col('valuenum').is_between(reference_outliers[variable]['min'],
                                                     reference_outliers[variable]['max'])

            df = df.filter((matches_variable & in_range) | ~matches_variable)
            continue

        elif is_drug:
            df = (
                df
                .with_columns([
                    pl.when(matches_variable)
                    .then(pl.col('bolus').clip(reference_outliers[variable]['bolus']['min'],
                                               reference_outliers[variable]['bolus']['max']))
                    .otherwise(pl.col('bolus')),

                    pl.when(matches_variable)
                    .then(pl.col('rate').clip(reference_outliers[variable]['rate']['min'],
                                              reference_outliers[variable]['rate']['max']))
                    .otherwise(pl.col('rate')),
                ])
            )

        else:
            raise ValueError(f'Variable {variable} not found in known variables')

    return df


def resize_hdf5_datasets(h5_array: h5py.File, next_size: int, label_features: list = None, context_window: int = 400):
    for embedding in ['features', 'timepoints', 'values', 'delta_time', 'delta_value']:
        for i in ['', 'next_']:
            h5_array[i + embedding].resize((next_size, context_window, 1))

    h5_array['labels'].resize((next_size, len(label_features)))

    return h5_array


def reverse_ids_for_sicdb(df):
    """
    To allow compatibility of code for both MIMIC and SICDB, sometimes we need to relabel "CaseID" as "subject_id".
    """
    if 'subject_id_original' in df.lazy().collect_schema().names():
        df = df.rename({'subject_id_original': 'subject_id', 'subject_id': 'CaseID'})
    else:
        df = df.rename({'subject_id': 'subject_id_original', 'CaseID': 'subject_id'})

    return df


def save_patient_ids_for_mimic(train_patient_ids, val_patient_ids, test_patient_ids):
    patient_ids_dir = './data/mimic/patient_ids'
    if not os.path.exists(patient_ids_dir):
        os.makedirs(patient_ids_dir, exist_ok=True)

    np.save(os.path.join(patient_ids_dir, 'train_patient_ids.npy'), train_patient_ids)
    np.save(os.path.join(patient_ids_dir, 'val_patient_ids.npy'), val_patient_ids)
    np.save(os.path.join(patient_ids_dir, 'test_patient_ids.npy'), test_patient_ids)

    return


def separate_basal_bolus_in_mimic(df):
    """
    We need to separate out 'rates' from 'boluses'.

    To do this, we follow the following rules:

       1) Remove any rows where the endtime is before / equal to the starttime
       2) For doses given over 1 minute, fill in any missing amount data, and remove rate
       3) For doses given over >1 minute, fill in any missing rate data, and then remove amount
       4) Rename amount to bolus

    """

    # 1) Remove rows where the endtime is before / equal to the starttime - hopefully shouldn't apply!
    df = df.filter((pl.col('endtime') > pl.col('starttime'))
                   | (pl.col('endtime').is_null()))

    df_no_rates = df.filter(pl.col('valuenum').is_not_null())
    df = df.filter(pl.col('valuenum').is_null())

    # 2) For all Med Boluses, remove the rate, and any rows where amount = 0
    criterion_1 = pl.col('ordercategorydescription').is_in(['Drug Push', 'Bolus'])
    criterion_2 = pl.col('amount') > 0

    df = (
        # Remove boluses equal to zero
        df.filter(
            # Is med bolus with amount over zero
            (criterion_1 & criterion_2)
            # Or is not med bolus i.e., is continuous rate
            | (~criterion_1)
        )
        # And remove rate and original rate from the boluses
        .with_columns([
            pl.when(criterion_1)
            .then(pl.lit(None).alias('rate'))
            .otherwise(pl.col('rate')),

            pl.when(criterion_1)
            .then(pl.lit(None).alias('rateuom'))
            .otherwise(pl.col('rateuom'))
        ])
    )

    # 3) For all Continuous Meds, remove the amount and remove any rows where rate = 0 (should be none)
    criterion_1 = pl.col('ordercategorydescription').is_in(['Continuous Med', 'Continuous IV'])
    criterion_2 = pl.col('amount') > 0

    df = (
        # Remove infusions with rate equal to zero
        df.filter(
            # Is infusion with amount over zero
            (criterion_1 & criterion_2)
            # Or is not infusion i.e., is med bolus
            | (~criterion_1)
        )
        # For infusions, convert amount to per-minute rate and then remove amount - currently independent of weight
        .with_columns([
            pl.when(criterion_1)
            .then(pl.col('amount').alias('rate') / (pl.col('endtime') - pl.col('starttime')).dt.total_minutes())
            .otherwise(pl.col('rate')),

            pl.when(criterion_1)
            .then((pl.col('amountuom').cast(pl.Utf8) + "/minute").cast(pl.Categorical('lexical')).alias('rateuom'))
            .otherwise(pl.col('rateuom'))
        ])

        # And remove amount/amountuom from the infusions
        .with_columns([
            pl.when(criterion_1)
            .then(pl.lit(None).alias('amount'))
            .otherwise(pl.col('amount'))])
        .unique()
    )

    # Merge back together
    df = pl.concat([df, df_no_rates])

    # 5) Rename 'amount' to 'bolus', and 'amountuom' to 'bolusuom'
    df = df.rename({'amount': 'bolus', 'amountuom': 'bolusuom'})

    return df


def separate_basal_bolus_in_sicdb(df):
    """
    Much simpler script than for MIMIC - SICDB is already separated out using IsSingleDose,
    so all we need to do is move 'amount' to 'bolus', and convert 'amount' to rate.
    """

    # 1) Remove all amounts = 0 for the drugs (boluses and rates)
    df = (
        df
        .filter((pl.col('amount') > 0) | pl.col('amount').is_null())
    )

    # 2) For all Med Boluses, remove rate from the boluses
    criteria = pl.col('IsSingleDose') == 1

    df = (
        df
        # Remove rate from the boluses, and add 'amountuom' and 'rateuom' as new columns
        .with_columns([
            pl.when(criteria)
            .then(pl.lit(None).alias('rate'))
            .otherwise(pl.col('rate')),

            pl.when(criteria)
            .then(pl.col('valueuom').alias('amountuom'))
            .otherwise(pl.lit(None).alias('amountuom')),

            # Create an empty / blank column for rateuom, to be filled in Step 3
            pl.lit(None).alias('rateuom')
        ])
        .with_columns([
            # valueuom can be set to Null for these rows now that we have amountuom
            pl.when(criteria)
            .then(pl.lit(None).alias('valueuom'))
            .otherwise(pl.col('valueuom')),
        ])
    )

    # 3) For all Continuous Meds, convert amount to rate
    criteria = pl.col('IsSingleDose') == 0

    df = (
        df
        # For infusions, convert amount to per-minute rate and then remove amount - currently independent of weight
        .with_columns([
            pl.when(criteria)
            .then(pl.col('amount').alias('rate') / (pl.col('endtime') - pl.col('starttime')).dt.total_minutes())
            .otherwise(pl.col('rate')),

            pl.when(criteria)
            .then((pl.col('valueuom').cast(pl.Utf8) + "/minute").cast(pl.Categorical('lexical')).alias('rateuom'))
            .otherwise(pl.col('rateuom'))
        ])
        .with_columns([
            # We can remove 'amount' and 'valueuom' for these rows, now that we have 'rate' and 'rateuom'
            pl.when(criteria)
            .then(pl.lit(None).alias('valueuom'))
            .otherwise(pl.col('valueuom')),

            pl.when(criteria)
            .then(pl.lit(None).alias('amount'))
            .otherwise(pl.col('amount'))
        ])
        .unique()
    )

    # 4) Rename 'amount' to 'bolus', and 'amountuom' to 'bolusuom'
    df = df.rename({'amount': 'bolus', 'amountuom': 'bolusuom'})

    df = df.select(['subject_id', 'CaseID', 'label', 'starttime', 'endtime', 'valuenum', 'valueuom', 'bolus',
                    'bolusuom', 'rate', 'rateuom', 'orderid', 'LaboratoryType'])

    return df


def split_labels_to_rate_and_bolus(df, *args):
    """
    We want a single 'value' column - to facilitate this for drugs, we create separate labels for 'rate' and 'bolus'
    """

    df = (
        df
        .with_columns([
            # Rename feature for boluses to feature + ' bolus'
            pl.when(pl.col('bolus').is_not_null())
            .then((pl.col('feature').cast(pl.Utf8) + ' bolus').cast(pl.Categorical('lexical')).alias('feature'))

            # Rename feature for original rates to feature + ' rate'
            .otherwise(pl.when(pl.col('rate').is_not_null())
                       .then(
                (pl.col('feature').cast(pl.Utf8) + ' rate').cast(pl.Categorical('lexical')).alias('feature'))

                       # Keep feature for valuenum as-is
                       .otherwise(pl.col('feature'))),
        ])
        .select([
                    'subject_id', 'feature', 'starttime',
                    # Move 'bolus' into 'value' for boluses
                    pl.when(pl.col('bolus').is_not_null())
                .then(pl.col('bolus').alias('value'))

                # Convert 'rate' into 'value' for rates
                .otherwise(pl.when(pl.col('rate').is_not_null())
                           .then(pl.col('rate').alias('value'))

                           # Keep 'valuenum' as 'value' for all other features
                           .otherwise(pl.col('valuenum')))] + [arg for arg in args])
        .unique()
    )
    return df
