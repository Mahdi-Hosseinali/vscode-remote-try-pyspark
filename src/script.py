
import datetime
import logging
from pathlib import Path
import sys

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


def reader(path):
    files = Path(path).glob('*.csv')
    for f in files:
        df = spark.read.options(header='True', inferSchema='True').csv(str(f))
        df = df.withColumn('Date', to_date_('Date'))
        logger.info(f'Read dataframe with {df.count()} rows')
        yield df

def to_date_(col):
    '''Convert multiple date formats from string'''

    formats = [
        'yyyy-M-d',
        'yyyy M d',
        'M/dd/yyyy',
        'yyyy MMM d',
        'M-d-yyyy',
    ]
    return F.coalesce(*[F.to_date(col, f) for f in formats])


def to_friday(df, dt):
    '''Convert all days to Fridays

    Note that this is not forward mapping as Saturday and Sunday are mapped backward.
    This is fine since all the data are supposed to be in "business days"
    To be consistent over the weekends, use the timestamps'''
    cols = df.columns
    # Convert all days to Friday of the week
    df = df.withColumn('friday', F.date_add(F.date_trunc('week', dt), 4))
    # Keep only the last record in each week
    w = Window.partitionBy('friday').orderBy(F.col(dt).desc())
    df = df.withColumn('rn', F.row_number().over(w)).where(F.col('rn') == 1)
    df = df.drop(dt).withColumnRenamed('friday', dt)
    return df.select(*cols)


def resample(df, dt, leap=60 * 60 * 24 * 7):
    '''Re-sample the data to 7 days intervals
    
    Generate the dates in epoch time, merge with the dataframe,
    then convert the date from the epoch times.'''
    def friday_timestamp(dt):
        diff = 4 - dt.weekday()
        if diff < 0:
            diff = 7 + diff
        friday = dt + datetime.timedelta(days=diff)
        return friday.timestamp()
    
    ts = 'temp_time'
    df = df.withColumn(ts, F.col(dt).cast('timestamp'))
    
    first_day, last_day = df.select(F.min(ts), F.max(ts)).first()
    ref = spark.range(
        friday_timestamp(first_day), friday_timestamp(last_day) + 1, leap
    ).toDF(ts).withColumn(ts, F.col(ts).cast('timestamp'))
    df = ref.join(df, ts, 'left').orderBy(ts)
    df = df.withColumn(dt, F.col(ts).cast('date'))
    return df.drop(ts)


def forward_fill(df, order, col):

    # needs some sort of partitioning to avoid collecting all data
    # Depending on the data the window size can be reduced to avoid collecting
    w = Window.orderBy(order).rowsBetween(-sys.maxsize, 0)
    return df.withColumn(col, F.last(col, ignorenulls=True).over(w))


def merge_all(dfs):
    ans = None
    marker = float('inf')
    logger.info('Reading and joining all files')
    for i, df in enumerate(dfs):
        # replace input nulls to something that we recognize
        # This is assuming the numbers are not inf, similar to the example
        df = df.na.fill(marker)
        df = to_friday(df, 'Date')
        if ans is None:
            ans = df
        else:
            ans = ans.join(df, 'Date', 'outer')
    logger.info(f'The shape is {ans.count()} x {len(ans.columns)}')
    logger.info('Resampling for all Fridays')
    ans = resample(ans, 'Date')
    cols = [i for i in ans.columns if i != 'Date']
    logger.info('forward filling')
    for c in cols:
        ans = forward_fill(ans, 'Date', c)
    # Replace back the original nulls and anything forward filled with them
    ans = ans.replace(marker, None)
    return ans

if __name__ == '__main__':
    path = 'data'
    dfs = reader(path)
    ans = merge_all(dfs)
    ans.write.csv('data/merged')