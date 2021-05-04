
import datetime
import random

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import pytest

from src.script import to_date_, to_friday, resample, forward_fill


def test_to_date_(get_spark_session):
    spark = get_spark_session
    in_dates = [
        ['2021-05-02'],
        ['2021 05 02'],
        ['05/02/2021'],
        ['2021 May 02'],
        ['05-02-2021'],
    ]

    df = spark.createDataFrame(in_dates, ('Date',))
    df_res = df.withColumn('Date', to_date_('Date'))
    df_res = df_res.toPandas()['Date'].unique()

    assert(len(df_res) == 1)
    if len(df_res) == 1:
        assert(df_res[0] == datetime.date(2021, 5, 2))


def test_to_friday(get_spark_session):
    spark = get_spark_session
    start = datetime.date(2021, 5, 3)
    friday = datetime.date(2021, 5, 7)
    one = datetime.timedelta(days=1)
    a_week = [(start + one, random.random()) for i in range(7)]

    df = spark.createDataFrame(a_week, ('Date','vals'))
    df_res = to_friday(df, 'Date')
    df_res = df_res.toPandas()['Date'].unique()

    assert(len(df_res) == 1)
    if len(df_res) == 1:
        assert(df_res[0] == friday)


@pytest.mark.parametrize('freq', ['B', 'W', 'M', 'BM', '2W-FRI', '2W-MON'])
def test_resample(get_spark_session, freq):
    '''Generate date ranges and sample them on Fridays'''
    def prepare_pdf(df):
        # pyspark ignores pandas index, 
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df['Date'] = df['Date'].dt.date

    spark = get_spark_session

    # resample works only when start and end are on Friday
    start = datetime.date(2021, 5, 7)
    end = datetime.date(2022, 5, 6)
    date_range = pd.date_range(start, end, freq=freq)
    pdf = pd.DataFrame(date_range.dayofyear, index=date_range)

    exp_df = pdf.resample('W-FRI').last()
    prepare_pdf(exp_df)

    prepare_pdf(pdf)
    df = spark.createDataFrame(pdf)

    df_res = resample(df, 'Date').toPandas()

    assert(exp_df['Date'].equals(df_res['Date']))


def test_forward_fill(get_spark_session):
    spark = get_spark_session

    pdf = pd.DataFrame({
        'val': [i for i in range(10)],
        'Date': pd.date_range(datetime.date(2021, 5, 7), freq='D', periods=10),
    })

    pdf.loc[::2, 'val'] = np.nan
    df = spark.createDataFrame(pdf)
    df_res = forward_fill(df, 'Date', 'val')

    pdf_filled = pdf.fillna(method='ffill')

    assert(pdf_filled.equals(df_res.toPandas()))