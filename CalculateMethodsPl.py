import polars as pl

"""
    ========================
        中金高频因子手册    
    ========================
"""


# 动量反转

def cal_mmt_pm(df: pl.DataFrame):
    """
    下午盘动量
    仅使用下午动量
    """
    return (
        df.filter(pl.col('time').is_in([130000000, 145900000]))
        .sort(by=['code', 'date', 'time'])
        .group_by(['code', 'date']).agg(
            (pl.col('close').last() / pl.col('open').first())
            .alias('mmt_pm')
        )
    )


def cal_mmt_last30(df: pl.DataFrame):
    """
    尾盘半小时动量
    仅使用尾盘30分钟动量
    """
    return (
        df.filter(pl.col('time').is_in([143000000, 145900000]))
        .sort(by=['code', 'date', 'time'])
        .group_by(['code', 'date']).agg(
            (pl.col('close').last() / pl.col('open').first())
            .alias('mmt_last30')
        )
    )


def cal_mmt_paratio(df: pl.DataFrame):
    """
    上下午盘动量差
    上下午盘动量差
    """
    return (
        df.lazy().with_columns(
            pl.when(pl.col('time') <= 113000000)
            .then(0)
            .otherwise(1)
            .alias('am_0_pm_1')
        ).group_by(['code', 'date', 'am_0_pm_1']).agg(
            (pl.col('close').last() / pl.col('open').first() - 1)
            .alias('mmt')
        ).group_by(['code', 'date']).agg(
            (pl.col('mmt').last() - pl.col('mmt').first())
            .alias('mmt_paratio')
        ).collect()
    )


def cal_mmt_am(df: pl.DataFrame):
    """
    上午盘动量
    仅使用上午盘动量
    """
    return (
        df.filter(pl.col('time').is_in([93000000, 112900000]))
        .sort(by=['code', 'date', 'time'])
        .group_by(['code', 'date']).agg(
            (pl.col('close').last() / pl.col('open').first())
            .alias('mmt_am')
        )
    )


def cal_mmt_between(df: pl.DataFrame):
    """
    去头尾动量
    使用剔除前后30分钟交易时间动量
    """
    return (
        df.filter(pl.col('time').is_in([100000000, 142900000]))
        .sort(by=['code', 'date', 'time'])
        .group_by(['code', 'date']).agg(
            (pl.col('close').last() / pl.col('open').first())
            .alias('mmt_between')
        )
    )


def cal_mmt_ols_qrs(df: pl.DataFrame):
    """
    分钟qrs指标
    50根分钟k线qrs指标
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.lazy().with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        )
        .select(['code', 'date', 'minute_in_trade', 'high', 'low'])
        .rolling(
            index_column='minute_in_trade',
            period='50i',
            group_by=['code', 'date']
        ).agg(
            pl.cov(a='low', b='high', ddof=0).alias('cov'),
            pl.var(column='low', ddof=0).alias('var_x'),
            pl.var(column='high', ddof=0).alias('var_y'),
            pl.col('high')
            .mean()
            .alias('mean_y'),
            pl.col('low')
            .mean()
            .alias('mean_x'),
            pl.len().alias('n')
        ).filter(pl.col('n') >= 50)
        .with_columns(
            pl.when(pl.col('var_x') != 0)
            .then(pl.col('cov') / pl.col('var_x'))
            .otherwise(pl.col('mean_y') / pl.col('mean_x'))
            .alias('beta'),
            pl.when(pl.col('var_y') * pl.col('var_x') != 0)
            .then(
                pl.col('cov').pow(0.5) / (pl.col('var_x') * pl.col('var_y'))
            )
            .otherwise(None)
            .alias('corr_square')
        )
        .group_by(['code', 'date'])
        .agg(
            pl.col('beta')
            .mean()
            .alias('beta_mean'),
            pl.col('beta')
            .std()
            .alias('beta_std'),
            pl.col('beta')
            .last()
            .alias('beta_last'),
            pl.col('corr_square')
            .mean()
            .alias('corr_square_mean'),
        ).select(
            pl.col('code'),
            pl.col('date'),
            pl.when(
                (
                    pl.col('beta_std') != 0
                ) & (
                    ~pl.col('corr_square_mean').is_null()
                )
            )
            .then(
                pl.col('corr_square_mean')
                * (pl.col('beta_last') - pl.col('beta_mean')) / pl.col('beta_std')
            )
            .otherwise(0)
            .alias('mmt_ols_qrs')
        ).collect()
    )


def cal_mmt_ols_corr_square_mean(df: pl.DataFrame):
    """
    分钟qrs衍生回归R方
    50根分钟k线最高价与最低价相关系数平方的均值
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.lazy()
        .with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        )
        .select(['code', 'date', 'minute_in_trade', 'high', 'low'])
        .rolling(
            index_column='minute_in_trade',
            period='50i',
            group_by=['code', 'date']
        ).agg(
            pl.cov(a='low', b='high', ddof=0).alias('cov'),
            pl.var(column='low', ddof=0).alias('var_x'),
            pl.var(column='high', ddof=0).alias('var_y'),
            pl.len().alias('n')
        )
        .filter(pl.col('n') >= 50)
        .with_columns(
            pl.when(pl.col('var_x') * pl.col('var_y') != 0)
            .then(
                pl.col('cov').pow(2) / (pl.col('var_x') * pl.col('var_y'))
            )
            .otherwise(None)
            .alias('corr_square')
        ).group_by(['code', 'date']).agg(
            pl.col('corr_square')
            .mean()
            .fill_null(0)
            .alias('mmt_ols_corr_square_mean')
        ).collect()
    )


def cal_mmt_ols_corr_mean(df: pl.DataFrame):
    """
    分钟qrs衍生相关系数均值
    50根分钟k线最高价与最低价相关系数的均值
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.lazy()
        .with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        )
        .select(['code', 'date', 'minute_in_trade', 'high', 'low'])
        .rolling(
            index_column='minute_in_trade',
            period='50i',
            group_by=['code', 'date']
        ).agg(
            pl.cov(a='low', b='high', ddof=0).alias('cov'),
            pl.var(column='low', ddof=0).alias('var_x'),
            pl.var(column='high', ddof=0).alias('var_y'),
            pl.len().alias('n')
        )
        .filter(pl.col('n') >= 50)
        .with_columns(
            pl.when(pl.col('var_x') * pl.col('var_y') != 0)
            .then(
                pl.col('cov') / (pl.col('var_x') * pl.col('var_y')).pow(0.5)
            )
            .otherwise(None)
            .alias('corr')
        ).group_by(['code', 'date']).agg(
            pl.col('corr')
            .mean()
            .fill_null(0)
            .alias('mmt_ols_corr_mean')
        ).collect()
    )


def cal_mmt_ols_beta_mean(df: pl.DataFrame):
    """
    分钟qrs衍生beta均值
    50根分钟k线最高价与最低价回归系数的均值
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.lazy().with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        )
        .select(['code', 'date', 'minute_in_trade', 'high', 'low'])
        .rolling(
            index_column='minute_in_trade',
            period='50i',
            group_by=['code', 'date']
        ).agg(
            pl.cov(a='low', b='high', ddof=0)
            .alias('cov'),
            pl.var(column='low', ddof=0)
            .alias('var_x'),
            pl.col('high')
            .mean()
            .alias('mean_y'),
            pl.col('low')
            .mean()
            .alias('mean_x'),
            pl.len()
            .alias('n')
        ).filter(pl.col('n') >= 50)
        .with_columns(
            pl.when(pl.col('var_x') != 0)
            .then(pl.col('cov') / pl.col('var_x'))
            .otherwise(pl.col('mean_y') / pl.col('mean_x'))
            .alias('beta')
        )
        .group_by(['code', 'date']).agg(
            pl.col('beta')
            .mean()
            .alias('mmt_ols_beta_mean')
        ).collect()
    )


def cal_mmt_ols_beta_zscore_last(df: pl.DataFrame):
    """
    分钟qrs衍生beta标准分
    50根分钟k线qrs指标
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.lazy().with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        ).select(['code', 'date', 'minute_in_trade', 'high', 'low'])
        .rolling(
            index_column='minute_in_trade',
            period='50i',
            group_by=['code', 'date']
        ).agg(
            pl.col('high')
            .mean()
            .alias('mean_y'),
            pl.col('low')
            .mean()
            .alias('mean_x'),
            pl.cov(a='low', b='high', ddof=0).alias('cov'),
            pl.var(column='low', ddof=0).alias('var_x'),
            pl.var(column='high', ddof=0).alias('var_y'),
            pl.len().alias('n')
        ).filter(pl.col('n') >= 50)
        .with_columns(
            pl.when(pl.col('var_x') != 0)
            .then(pl.col('cov') / pl.col('var_x'))
            .otherwise(pl.col('mean_y') / pl.col('mean_x'))
            .alias('beta')
        ).group_by(['code', 'date']).agg(
            pl.when(pl.col('beta').std() > 0)
            .then(
                (pl.col('beta').last() - pl.col('beta').mean()) / pl.col('beta').std()
            )
            .otherwise(pl.col('beta').mean())
            .alias('mmt_ols_beta_zscore_last')
        ).collect()
    )


def cal_mmt_top50VolumeRet(df: pl.DataFrame):
    """
    50顶量成交动量
    成交量最高50根k线成交量收益率动量
    """
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open'))
            .alias('ret')
        )
        .filter(
            (
                pl.col('volume') >= (
                    pl.col('volume')
                    .top_k(50)
                    .min()
                )
            ).over(['code', 'date'])
        )
        .group_by(['code', 'date']).agg(
            (pl.col('ret').product() - 1)
            .alias('mmt_top50VolumeRet')
        ).collect()
    )


def cal_mmt_bottom50VolumeRet(df: pl.DataFrame):
    """
    50底量成交动量
    最低成交量的50根k线收益率动量
    """
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open'))
            .alias('ret')
        )
        .filter(
            (
                pl.col('volume') <= (
                    pl.col('volume')
                    .bottom_k(50)
                    .max()
                )
            ).over(['code', 'date'])
        )
        .group_by(['code', 'date']).agg(
            (pl.col('ret').product() - 1)
            .alias('mmt_bottom50VolumeRet')
        ).collect()
    )


def cal_mmt_top20VolumeRet(df: pl.DataFrame):
    """
    20顶量成交动量
    成交量最高20根k线成交量收益率动量
    """
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open'))
            .alias('ret')
        )
        .filter(
            (
                pl.col('volume') >= (
                    pl.col('volume')
                    .top_k(20)
                    .min()
                )
            ).over(['code', 'date'])
        )
        .group_by(['code', 'date']).agg(
            (pl.col('ret').product() - 1)
            .alias('mmt_top20VolumeRet')
        ).collect()
    )


def cal_mmt_bottom20VolumeRet(df: pl.DataFrame):
    """
    20底量成交动量
    最低成交量的20根k线收益率动量
    """
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open'))
            .alias('ret')
        )
        .filter(
            (
                pl.col('volume') <= (
                    pl.col('volume')
                    .bottom_k(50)
                    .max()
                )
            ).over(['code', 'date'])
        )
        .group_by(['code', 'date']).agg(
            (pl.col('ret').product() - 1)
            .alias('mmt_bottom20VolumeRet')
        ).collect()
    )


# 波动率

def cal_vol_volume1min(df: pl.DataFrame):
    """
    分钟成交量的标准差
    日内分钟k线成交量的标准差
    """
    return (
        df.group_by(['code', 'date']).agg(
            pl.col('volume')
            .std()
            .alias('vol_volume1min')
        )
    )


def cal_vol_range1min(df: pl.DataFrame):
    """
    分钟极比的标准差
    分钟k线的最大值最小值比值的标准差
    """
    return (
        df.select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('high') / pl.col('low'))
            .alias('range')
        ).group_by(['code', 'date']).agg(
            pl.col('range')
            .std()
            .alias('vol_range1min')
        )
    )


def cal_vol_return1min(df: pl.DataFrame):
    """
    分钟收益率的标准差
    日内分钟收益率的标准差
    """
    return (
        df.select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('return')
        ).group_by(['code', 'date']).agg(
            pl.col('return')
            .std()
            .alias('vol_return1min')
        )
    )


def cal_vol_upVol(df: pl.DataFrame):
    """
    上行波动率
    使用日内分钟级别数据，计算分钟级上行波动率
    """
    return (
        df.select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('return')
        )
        .with_columns(
            pl.when(pl.col('return') > 0)
            .then(pl.col('return'))
            .otherwise(None)
            .alias('up_return')
        ).group_by(['code', 'date']).agg(
            pl.col('up_return')
            .std()
            .fill_null(0)
            .alias('vol_upVol')
        )
    )


def cal_vol_upRatio(df: pl.DataFrame):
    """
    上行波动率占比
    使用日内分钟级别数据，计算分钟级上行波动率占总波动的比例
    """
    return (
        df.select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('return')
        )
        .with_columns(
            pl.when(pl.col('return') > 0)
            .then(pl.col('return'))
            .otherwise(None)
            .alias('up_return')
        ).group_by(['code', 'date']).agg(
            (
                pl.col('up_return')
                .std()
                .fill_null(0) / pl.col('return').std()
            )
            .alias('vol_upRatio')
        )
    )


def cal_vol_downVol(df: pl.DataFrame):
    """
    下行波动率
    使用日内分钟级别数据，计算分钟级下行波动率
    """
    return (
        df.select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('return')
        )
        .with_columns(
            pl.when(pl.col('return') < 0)
            .then(pl.col('return'))
            .otherwise(None)
            .alias('down_return')
        ).group_by(['code', 'date']).agg(
            pl.col('down_return')
            .std()
            .fill_null(0)
            .alias('vol_downVol')
        )
    )


def cal_vol_downRatio(df: pl.DataFrame):
    """
    下行波动率占比
    使用日内分钟级别数据，计算分钟级下行波动率占总波动的比例
    """
    return (
        df.select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('return')
        )
        .with_columns(
            pl.when(pl.col('return') < 0)
            .then(pl.col('return'))
            .otherwise(None)
            .alias('down_return')
        ).group_by(['code', 'date']).agg(
            (
                pl.col('down_return')
                .std()
                .fill_null(0) / pl.col('return').std()
            )
            .alias('vol_downRatio')
        )
    )


# 高阶特征

def cal_shape_skew(df: pl.DataFrame):
    """
    分钟收益率偏度
    分钟k线收益率的偏度
    """
    shape_skew = df.group_by(['code', 'date']).agg(
        (pl.col('close') / pl.col('open') - 1)
        .skew()
        .alias('shape_skew')
    )
    return shape_skew


def cal_shape_kurt(df: pl.DataFrame):
    """
    分钟收益率峰度
    分钟k线收益率的峰度
    """
    shape_kurt = df.group_by(['code', 'date']).agg(
        (pl.col('close') / pl.col('open') - 1)
        .kurtosis()
        .alias('shape_kurt')
    )
    return shape_kurt


def cal_shape_skratio(df: pl.DataFrame):
    """
    分钟收益率峰度偏度比
    分钟k线收益率的峰度与偏度的比值
    """
    df = df.select(
        'code',
        'date',
        (pl.col('close') / pl.col('open') - 1).alias('return')
    )
    shape_skratio = df.group_by(['date', 'code']).agg(
        (pl.col('return').skew() / pl.col('return').kurtosis())
        .alias('shape_skratio')
    )
    return shape_skratio


def cal_shape_skewVol(df: pl.DataFrame):
    """
    分钟成交量占比的偏度
    分钟k线成交量占比的偏度
    """
    shape_skew_vol = df.group_by(['code', 'date']).agg(
        (pl.col('volume') / pl.col('volume').sum())
        .skew()
        .alias('shape_skewVol')
    )
    return shape_skew_vol


def cal_shape_kurtVol(df: pl.DataFrame):
    """
    分钟成交量占比的峰度
    分钟k线成交量占比的峰度
    """
    shape_kurt_vol = df.group_by(['code', 'date']).agg(
        (pl.col('volume') / pl.col('volume').sum())
        .kurtosis()
        .alias('shape_kurtVol')
    )
    return shape_kurt_vol


def cal_shape_skratioVol(df: pl.DataFrame):
    """
    分钟成交量占比峰度偏度比
    分钟k线成交量占比的峰度与偏度的比值
    """
    shape_skratio_vol = df.with_columns(
        (pl.col('volume') / pl.col('volume').sum())
        .over(['code', 'date'])
        .alias('volume_d')
    ).group_by(['code', 'date']).agg(
        (pl.col('volume_d').skew() / pl.col('volume_d').kurtosis())
        .alias('shape_skratioVol')
    )
    return shape_skratio_vol


# 流动性

def cal_liq_amihud_1min(df: pl.DataFrame):
    """
    Amihud非流动性因子
    计算日内分钟级别数据构建常见的Amihud非流动因子
    """
    liq_amihud_1min = (
        df.lazy().select(
            pl.col('code'),
            pl.col('date'),
            pl.col('volume')
            .fill_null(0),
            pl.col('close').pct_change()
            .over('code')
            .abs()
            .fill_null(0)
            .alias('pct_change_abs')
        ).with_columns(
            pl.when(pl.col('volume') > 0)
            .then(pl.col('pct_change_abs') / pl.col('volume'))
            .otherwise(0)
            .alias('amihud')
        ).group_by(['code', 'date']).agg(
            pl.col('amihud')
            .sum()
            .alias('liq_amihud_1min')
        ).collect()
    )
    return liq_amihud_1min


def cal_liq_closeprevol(df: pl.DataFrame):
    """
    集合竞价前成交量
    计算集合竞价前的成交量
    """
    liq_closeprevol = (
        df.filter(pl.col('time') < 145700000)
        .group_by(['code', 'date']).agg(
            pl.col('volume').sum().alias('liq_closeprevol')
        )
    )
    return liq_closeprevol


def cal_liq_closevol(df: pl.DataFrame):
    """
    收盘前3分钟成交量
    计算收盘前3分钟成交量
    """
    liq_closevol = (
        df.filter(pl.col('time') >= 145700000)
        .group_by(['code', 'date']).agg(
            pl.col('volume').sum().alias('liq_closevol')
        )
    )
    return liq_closevol


def cal_liq_firstCallR(df: pl.DataFrame):
    """
    开盘集合竞价成交量占比
    使用日内tick数据计算上午开盘9：25之前的集合竞价总交易量占全天交易量的比例
    改为使用分钟频数据计算
    """
    liq_first_call_r = df.group_by(['code', 'date']).agg(
        (pl.col('volume').first() / pl.col('volume').sum())
        .alias('liq_firstCallR')
    )
    return liq_first_call_r


def cal_liq_lastCallR(df: pl.DataFrame):
    """
    收盘集合竞价成交量占比
    使用日内tick数据计算下午收盘前14：57-15：00的集合竞价的总交易量占全天交易量的比例
    改为使用分钟频数据计算
    """
    liq_last_call_r = df.group_by(['code', 'date']).agg(
        (
            (
                pl.col("volume")
                .filter(pl.col("time") >= 145700000)
                .sum()
            ) / pl.col('volume').sum()
        ).alias('liq_lastCallR')
    )
    return liq_last_call_r


def cal_liq_openvol(df: pl.DataFrame):
    """
    开盘集合竞价成交量
    计算开盘集合竞价成交量
    """
    liq_openvol = df.group_by(['code', 'date']).agg(
        pl.col('volume').first().alias('liq_openvol')
    )
    return liq_openvol


# 量价相关性

def cal_corr_prv(df: pl.DataFrame):
    """
    计算分钟收益率与成交量相关系数
    计算分钟收益率与成交量相关系数
    """
    corr_prv = df.group_by(['code', 'date']).agg(
        pl.corr(
            pl.col('close').pct_change(),
            pl.col('volume')
        ).alias('corr_prv')
    )
    return corr_prv


def cal_corr_prvr(df: pl.DataFrame):
    """
    分钟收益率与成交量变化率相关系数
    计算分钟收益率与成交量变化率相关系数
    """
    corr_prvr = df.lazy().filter(
        pl.col('volume') != 0
    ).select(
        pl.col('code'),
        pl.col('date'),
        pl.col('close')
        .pct_change()
        .over('code')
        .alias('close_change'),
        pl.col('volume')
        .pct_change()
        .over('code')
        .alias('volume_change')
    ).group_by(['code', 'date']).agg(
        pl.corr(
            pl.col('close_change'),
            pl.col('volume_change')
        ).alias('corr_prvr')
    ).collect()
    return corr_prvr


def cal_corr_pv(df: pl.DataFrame):
    """
    分钟收盘价与成交量相关系数
    计算分钟收盘价与成交量相关系数
    """
    corr_pv = df.group_by(['code', 'date']).agg(
        pl.corr(
            pl.col('close'),
            pl.col('volume')
        ).alias('corr_pv')
    )
    return corr_pv


def cal_corr_pvd(df: pl.DataFrame):
    """
    分钟收盘价与滞后成交量相关系数
    计算分钟收盘价与滞后成交量相关系数
    """
    corr_pvd = df.group_by(['code', 'date']).agg(
        pl.corr(
            pl.col('close'),
            pl.col('volume').shift(1)
        ).alias('corr_pvd')
    )
    return corr_pvd


def cal_corr_pvl(df: pl.DataFrame):
    """
    分钟收盘价与领先成交量相关系数
    计算分钟收盘价与领先成交量相关系数
    """
    corr_pvl = df.group_by(['code', 'date']).agg(
        pl.corr(
            pl.col('close'),
            pl.col('volume').shift(-1)
        ).alias('corr_pvl')
    )
    return corr_pvl


def cal_corr_pvr(df: pl.DataFrame):
    """
    分钟收盘价与成交量变化率相关系数
    计算分钟收盘价与成交量变化率相关系数
    """
    corr_pvr = df.filter(
        pl.col('volume') != 0
    ).group_by(['code', 'date']).agg(
        pl.corr(
            pl.col('close'),
            pl.col('volume').pct_change()
        ).alias('corr_pvr')
    )
    return corr_pvr


# 筹码分布

def cal_doc_kurt(df: pl.DataFrame):
    """
    分钟收益率分组筹码峰度
    计算分钟级k线数据按照收益率分布成交量分布的峰度
    """
    doc_kurt = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .alias("return")
        ).group_by(["code", "date", "return"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col("volume_d")
            .kurtosis()
            .alias("doc_kurt")
        )
    )
    return doc_kurt


def cal_doc_skew(df: pl.DataFrame):
    """
    分钟收益率分组筹码偏度
    计算分钟级k线数据按照收益率分布成交量分布的偏度
    """
    doc_skew = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .alias("return")
        ).group_by(["code", "date", "return"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col("volume_d")
            .skew()
            .alias("doc_skew")
        )
    )
    return doc_skew


def cal_doc_std(df: pl.DataFrame):
    """
    分钟收益率分组筹码标准差
    计算分钟级k线数据按照收益率分布成交量分布的标准差
    """
    doc_std = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .alias("return")
        ).group_by(["code", "date", "return"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col("volume_d")
            .skew()
            .alias("doc_std")
        )
    )
    return doc_std


def cal_doc_pdf60(df: pl.DataFrame):
    """
    分钟收益率分组筹码60%占比收益率分位
    计算分钟收益率分组筹码60%占比收益率分位
    """
    doc_pdf60 = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .rank()
            .alias("return_rank")
        ).group_by(["code", "date", "return_rank"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col('return_rank').filter(
                pl.col("volume_d")
                .cum_sum() > 0.6
            ).sort()
            .first()
            .alias('doc_pdf60')
        )
    )
    return doc_pdf60


def cal_doc_pdf70(df: pl.DataFrame):
    """
    分钟收益率分组筹码70%占比收益率分位
    计算分钟收益率分组筹码70%占比收益率分位
    """
    doc_pdf70 = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .rank()
            .alias("return_rank")
        ).group_by(["code", "date", "return_rank"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col('return_rank').filter(
                pl.col("volume_d")
                .cum_sum() > 0.7
            ).sort()
            .first()
            .alias('doc_pdf70')
        )
    )
    return doc_pdf70


def cal_doc_pdf80(df: pl.DataFrame):
    """
    分钟收益率分组筹码80%占比收益率分位
    计算分钟收益率分组筹码80%占比收益率分位
    """
    doc_pdf80 = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .rank()
            .alias("return_rank")
        ).group_by(["code", "date", "return_rank"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col('return_rank').filter(
                pl.col("volume_d")
                .cum_sum() > 0.8
            ).sort()
            .first()
            .alias('doc_pdf80')
        )
    )
    return doc_pdf80


def cal_doc_pdf90(df: pl.DataFrame):
    """
    分钟收益率分组筹码90%占比收益率分位
    计算分钟收益率分组筹码90%占比收益率分位
    """
    doc_pdf90 = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .rank()
            .alias("return_rank")
        ).group_by(["code", "date", "return_rank"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col('return_rank').filter(
                pl.col("volume_d")
                .cum_sum() > 0.9
            ).sort()
            .first()
            .alias('doc_pdf90')
        )
    )
    return doc_pdf90


def cal_doc_pdf95(df: pl.DataFrame):
    """
    分钟收益率分组筹码95%占比收益率分位
    计算分钟收益率分组筹码95%占比收益率分位
    """
    doc_pdf95 = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .rank()
            .alias("return_rank")
        ).group_by(["code", "date", "return_rank"]).agg(
            pl.col("volume_d")
            .sum()
        ).group_by(["code", "date"]).agg(
            pl.col('return_rank').filter(
                pl.col("volume_d")
                .cum_sum() > 0.95
            ).sort()
            .first()
            .alias('doc_pdf95')
        )
    )
    return doc_pdf95


def cal_doc_vol10_ratio(df: pl.DataFrame):
    """
    分钟收益率分组筹码前10大占比
    计算分钟收益率分组筹码前10大占比
    """
    doc_vol10_ratio = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .alias("return")
        ).group_by(['code', 'date']).agg(
            pl.col('volume_d')
            .top_k(10)
            .sum()
            .alias('doc_vol10_ratio')
        )
    )
    return doc_vol10_ratio


def cal_doc_vol5_ratio(df: pl.DataFrame):
    """
    分钟收益率分组筹码前5大占比
    计算分钟收益率分组筹码前5大占比
    """
    doc_vol5_ratio = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .alias("return")
        ).group_by(['code', 'date']).agg(
            pl.col('volume_d')
            .top_k(5)
            .sum()
            .alias('doc_vol5_ratio')
        )
    )
    return doc_vol5_ratio


def cal_doc_vol50_ratio(df: pl.DataFrame):
    """
    分钟收益率分组筹码前50大占比
    计算分钟收益率分组筹码前50大占比
    """
    doc_vol50_ratio = (
        df.with_columns(
            (pl.col("volume") / pl.col("volume").sum().over(["code", "date"]))
            .alias("volume_d"),
            (pl.col("close").last().over(["code", "date"]) / pl.col("close"))
            .alias("return")
        ).group_by(['code', 'date']).agg(
            pl.col('volume_d')
            .top_k(5)
            .sum()
            .alias('doc_vol50_ratio')
        )
    )
    return doc_vol50_ratio

# 资金成交


def cal_trade_bottom20retRatio(df: pl.DataFrame):
    """
    后20k线收益率成交占比
    后20根k线每根的收益率乘以其成交量所占比例，得到的加权收益率之和
    """
    trade_bottom20retRatio = (
        df.lazy().filter(pl.col('time') >= 144000000)
        .with_columns(
            (pl.col('close') / pl.col('open') - 1)
            .alias('ret'),
            (pl.col('volume') / (pl.col('volume').sum().over('code') + 1))
            .alias('volume_d')
        ).group_by(['code', 'date']).agg(
            (pl.col('volume_d') * pl.col('ret'))
            .sum()
            .alias('trade_bottom20retRatio')
        ).collect()
    )
    return trade_bottom20retRatio


def cal_trade_bottom50retRatio(df: pl.DataFrame):
    """
    后50k线收益率成交占比
    后50根k线每根的收益率乘以其成交量所占比例，得到的加权收益率之和
    """
    trade_bottom50retRatio = (
        df.lazy().filter(pl.col('time') >= 141000000)
        .with_columns(
            (pl.col('close') / pl.col('open') - 1)
            .alias('ret'),
            (pl.col('volume') / (
                pl.when(pl.col('volume').sum().over('code') == 0)
                .then(1)
                .otherwise(pl.col('volume').sum().over('code'))
            )).alias('volume_d')
        ).group_by(['code', 'date']).agg(
            (pl.col('volume_d') * pl.col('ret'))
            .sum()
            .alias('trade_bottom50retRatio')
        ).collect()
    )
    return trade_bottom50retRatio


def cal_trade_headRatio(df: pl.DataFrame):
    """
    开盘成交占比
    开盘一定时间内的成交量与当日总成交的比例
    """
    trade_headRatio = (
        df.lazy().with_columns(
            pl.when(pl.col('time') <= 100000000)
            .then(pl.col('volume'))
            .otherwise(0)
            .alias('headVolume')
        )
        .group_by(['code', 'date']).agg(
            pl.col('headVolume')
            .sum(),
            pl.col('volume')
            .sum()
        ).select(
            pl.col('code'),
            pl.col('date'),
            pl.when(pl.col('volume') > 0)
            .then(pl.col('headVolume') / pl.col('volume'))
            .otherwise(0.125)
            .alias('trade_headRatio')
        ).collect()
    )
    return trade_headRatio


def cal_trade_tailRatio(df: pl.DataFrame):
    """
    尾盘成交占比
    收盘前一定时间内的成交量与当日总成交的比例
    """
    trade_tailRatio = (
        df.lazy().with_columns(
            pl.when(pl.col('time') >= 143000000)
            .then(pl.col('volume'))
            .otherwise(0)
            .alias('tailVolume')
        )
        .group_by(['code', 'date']).agg(
            pl.col('tailVolume')
            .sum(),
            pl.col('volume')
            .sum()
        ).select(
            pl.col('code'),
            pl.col('date'),
            pl.when(pl.col('volume') > 0)
            .then(pl.col('tailVolume') / pl.col('volume'))
            .otherwise(0.125)
            .alias('trade_tailRatio')
        ).collect()
    )
    return trade_tailRatio


def cal_trade_top20retRatio(df: pl.DataFrame):
    """
    前20K线收益率成交占比
    前20根K线中，收益率与成交量比例的均值
    """
    trade_top20retRatio = (
        df.filter(pl.col('time') <= 95000000)
        .with_columns(
            (pl.col('volume') / pl.col('volume').sum().over(['code', 'date']))
            .alias('volume_d'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('pct_change')
        )
        .group_by(['code', 'date']).agg(
            (pl.col('pct_change') / pl.col('volume_d'))
            .mean()
            .alias('trade_top20retRatio')
        )
    )
    return trade_top20retRatio


def cal_trade_top50retRatio(df: pl.DataFrame):
    """
    前50K线收益率成交占比
    前50根K线中，收益率与成交量比例的均值
    """
    trade_top50retRatio = (
        df.filter(pl.col('time') <= 102000000)
        .with_columns(
            (pl.col('volume') / pl.col('volume').sum().over(['code', 'date']))
            .alias('volume_d'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('pct_change')
        )
        .group_by(['code', 'date']).agg(
            (pl.col('pct_change') / pl.col('volume_d'))
            .mean()
            .alias('trade_top50retRatio')
        )
    )
    return trade_top50retRatio


def cal_trade_topNeg20retRatio(df: pl.DataFrame):
    """
    前20K线下跌收益率成交占比
    前20根K线中，收益率为负的绝对平均收益率与成交量比例的均值
    """
    trade_topNeg20retRatio = (
        df.filter(pl.col('time') <= 95000000)
        .with_columns(
            (pl.col('volume') / pl.col('volume').sum().over(['code', 'date']))
            .alias('volume_d'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('pct_change')
        )
        .group_by(['code', 'date']).agg(
            (
                (
                    pl.when(pl.col('pct_change') < 0)
                    .then(pl.col('pct_change').abs())
                    .otherwise(0)
                ) / pl.col('volume_d')
            )
            .mean()
            .alias('trade_topNeg20retRatio')
        )
    )
    return trade_topNeg20retRatio


def cal_trade_topPos20retRatio(df: pl.DataFrame):
    """
    前20K线上涨收益率成交占比
    前20根K线中，收益率为正的平均收益率与成交量比例的均值
    """
    trade_topPos20retRatio = (
        df.filter(pl.col('time') <= 95000000)
        .with_columns(
            (pl.col('volume') / pl.col('volume').sum().over(['code', 'date']))
            .alias('volume_d'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('pct_change')
        )
        .group_by(['code', 'date']).agg(
            (
                (
                    pl.when(pl.col('pct_change') > 0)
                    .then(pl.col('pct_change').abs())
                    .otherwise(0)
                ) / pl.col('volume_d')
            )
            .mean()
            .alias('trade_topPos20retRatio')
        )
    )
    return trade_topPos20retRatio
