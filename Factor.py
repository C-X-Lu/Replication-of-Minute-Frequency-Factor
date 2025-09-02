import polars as pl
import os
from typing import Literal
import tempfile
import matplotlib.pyplot as plt

class Factor:
    def __init__(self, factor_name: str, factor_exposure: pl.DataFrame=None):
        """
        因子类
        :param factor_name: str 因子的名字
        :param factor_exposure: pl.DataFrame 因子暴露
        """
        self.factor_name = factor_name
        self.factor_exposure = factor_exposure
        self.IC = None
        self.ICIR = None
        self.rank_IC = None
        self.rank_ICIR = None

    @staticmethod
    def _read_daily_pv_data(column_need: str|list[str]=None) -> pl.DataFrame:
        """
        读取日频量价数据，数据包含：code/date/
        open/high/low/close/close_adjust/pct_change/
        volume/amount/
        cmc/tmc/
        limit_down/limit_up
        :param column_need: 需要的列
        :return:
        """
        column_dict = {
            'Trddt': 'date',
            'Stkcd': 'code',
            'Opnprc': 'open',
            'Hiprc': 'high',
            'Loprc': 'low',
            'Clsprc': 'close',
            'Dnshrtrd': 'volume',
            'Dnvaltrd': 'amount',
            'ChangeRatio': 'pct_change',
            'Dsmvosd': 'cmc',
            'Dsmvtll': 'tmc',
            'Adjprcwd': 'close_adjust',
            'LimitDown': 'limit_down',
            'LimitUp': 'limit_up'
        }
        pv_data = (
            pl.scan_parquet(r'D:\QuantData\Price_Volume.parquet')
            .with_columns(
                pl.col('Trddt')
                .str.to_date(format='%Y-%m-%d')
            ).rename(column_dict)
        )
        if column_need is None:
            column_need = column_dict.keys()
        pv_data = pv_data.select(
            pl.col(column)
            for column in column_need
            if column in pv_data.collect_schema().names()
        ).collect()
        return pv_data

    def to_parquet(self, path: str=None):
        r"""
        将数据保存为parquet。
        :param path: 保存的路径, 默认路径为'D:\QuantData\MinuteFreqFactor', 默认名称为因子名。
        """
        if path is None:
            path = r'D:\QuantData\MinuteFreqFactor'
        if not path.endswith('.parquet'):
            path = os.path.join(path, f'{self.factor_name}.parquet')

        temp_dir = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(
                dir=temp_dir, delete=False, suffix='.parquet'
        ) as tmp:
            temp_path = tmp.name

        try:
            self.factor_exposure.write_parquet(temp_path)
            # 替换原文件
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def coverage(self, plot_out=True, return_df=False) -> pl.DataFrame | None:
        """
        计算因子覆盖度
        :param plot_out: 是否输出每日因子暴露有效数量图，默认为False
        :param return_df: 是否返回包含每日因子暴露有效数量的DataFrame，默认为False
        :return:
        """
        coverage = (
            self.factor_exposure.filter(
                ~pl.col(self.factor_name).is_nan()
            ).group_by('date').agg(
                pl.col(self.factor_name).count()
            ).sort(by='date')
        )
        if plot_out:
            color = 'tab:blue'
            plt.figure(figsize=(12, 8))
            plt.bar(
                coverage['date'], coverage[self.factor_name],
                color=color, alpha=0.6, label=f'{self.factor_name} coverage'
            )
            if coverage.shape[0] > 20:
                n = max(1, len(coverage) // 10)
                plt.xticks(coverage['date'][::n], rotation=45)
            else:
                plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            plt.title('coverage plot')
            plt.tight_layout()
            plt.show()
        if return_df:
            return coverage
        return None

    def ic_test(
            self,
            future_days: int=5,
            plot_out: bool=True,
            plot_variable: str='IC',
            return_df: bool=False
    ) -> pl.DataFrame|None:
        """
        因子IC与rank_IC测试
        :param future_days: 与未来多少天的收益计算相关系数
        :param plot_out: 是否输出IC与rankIC的累计图，默认为False
        :param plot_variable: 输出IC图还是rank_IC图，默认为IC
        :param return_df: 是否返回包含每日IC与rank_IC的DataFrame，默认为False
        :return:
        """
        pv_data = (
            self._read_daily_pv_data(['code', 'date', 'pct_change'])
            .lazy().sort(by=['code', 'date'])
            .with_columns(
                (
                    (pl.col('pct_change') + 1)
                    .log()
                    .rolling_sum(future_days, min_samples=future_days)
                    .over('code')
                    .exp() - 1
                )
                .alias('rolling_change')
            ).select(
                pl.col('code'),
                pl.col('date'),
                pl.col('rolling_change')
                .shift(-future_days)
                .over('code')
                .alias('future_return')
            ).collect()
        )
        ic_df = (
            pl.concat(
                items=[
                    self.factor_exposure
                    .filter(
                        ~pl.col(self.factor_name).is_nan()
                    ),
                    pv_data
                ], how='align_left'
            ).group_by('date').agg(
                pl.corr(
                    pl.col(self.factor_name),
                    pl.col('future_return'),
                    method='pearson'
                ).alias('IC'),
                pl.corr(
                    pl.col(self.factor_name),
                    pl.col('future_return'),
                    method='spearman'
                ).alias('rank_IC')
            ).filter(
                (~pl.col('IC').is_null()) & (~pl.col('IC').is_nan())
            ).sort(by='date')
        )
        self.IC = ic_df['IC'].mean()
        self.rank_IC = ic_df['rank_IC'].mean()
        self.ICIR = self.IC / ic_df['IC'].std()
        self.rank_ICIR = self.rank_IC / ic_df['rank_IC'].std()
        if plot_out:  # 输出图
            fig, ax1 = plt.subplots(figsize=(12, 6))

            color = 'tab:blue'
            ax1.set_xlabel('date')
            ax1.set_ylabel(ylabel=plot_variable, color=color)
            ax1.bar(
                ic_df['date'], ic_df[plot_variable],
                color=color, alpha=0.6, width=1.0
            )
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(ylabel=f'cum {plot_variable}', color=color)
            ax2.plot(
                ic_df['date'], ic_df[plot_variable].cum_sum(),
                color=color, linewidth=2.0, label=f'cum {plot_variable}'
            )
            ax2.tick_params(axis='y', labelcolor=color)

            ax1.grid(visible=True, linestyle='--', alpha=0.7)

            if ic_df.shape[0] > 20:
                n = max(1, len(ic_df) // 10)
                plt.xticks(ic_df['date'][::n], rotation=45)
            else:
                plt.xticks(rotation=45)

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best')

            plt.title(f'{plot_variable} plot')
            plt.tight_layout()
            plt.show()
        if return_df:  # 返回DataFrame
            return ic_df
        return None

    def group_test(
            self,
            frequency: Literal['weekly', 'monthly', 'quarterly', 'yearly'] = 'monthly',
            weight_param: Literal['tmc', 'cmc', None] = None,
            group_num: int = 5,
            plot_out: bool = True,
            return_df: bool = False
    ) -> pl.DataFrame | None:
        """
        因子分组测试
        :param frequency: 调仓频率，默认为月频
        :param weight_param: 加权方式，包括总市值tmc、流通市值cmc和等权，默认等权
        :param group_num: 分组数量，默认为5
        :param plot_out: 是否输出分组收益图，默认为True
        :param return_df: 是否输出DataFrame，默认为False
        :return:
        """
        if frequency == 'weekly':
            group_param = '1w'
        elif frequency == 'monthly':
            group_param = '1mo'
        elif frequency == 'quarterly':
            group_param = '1q'
        elif frequency == 'yearly':
            group_param = '1y'
        pv_data = (
            self._read_daily_pv_data(['code', 'date', 'pct_change', 'tmc', 'cmc'])
        )
        if weight_param is None:
            expr = (
                pl.col('pct_change')
                .mean()
            )
        elif weight_param == 'tmc':
            expr = pl.when(
                pl.col('tmc').sum() != 0
            ).then(
                ((pl.col('pct_change') * pl.col('tmc')).sum() / pl.col('tmc').sum())
                .alias('pct_change')
            ).otherwise(
                0
            )
        elif weight_param == 'cmc':
            expr = pl.when(
                pl.col('cmc').sum() != 0
            ).then(
                ((pl.col('pct_change') * pl.col('cmc')).sum() / pl.col('cmc').sum())
                .alias('pct_change')
            ).otherwise(0)
        group_df = (
            pl.concat(
                [self.factor_exposure, pv_data],
                how='align_left'
            ).lazy().with_columns(
                pl.col(self.factor_name)
                .qcut(
                    group_num,
                    labels=[f"group_{i+1}" for i in range(group_num)],
                    allow_duplicates=True
                )
                .over('date')
                .alias('group')
            ).group_by_dynamic(
                'date', every=group_param, label='right', group_by='code'
            ).agg(
                (
                    (
                        pl.col('pct_change') + 1
                    ).product() - 1
                ).alias('pct_change'),
                pl.col('group').last(),
                pl.col('tmc').last(),
                pl.col('cmc').last()
            ).sort(by=['date', 'group'])
            .with_columns(
                pl.col('group')
                .shift(1)
                .over('code'),
                pl.col('tmc')
                .shift(1)
                .over('code'),
                pl.col('cmc')
                .shift(1)
                .over('code')
            ).filter(
                ~pl.col('group').is_null()
            ).group_by(['date', 'group']).agg(
                expr
            ).sort(by=['date', 'group'])
            .collect()
        )
        if plot_out:  # 输出图
            plt.figure(figsize=(12, 8))
            for group in group_df['group'].unique().sort():
                plot_df = group_df.filter(
                    pl.col('group') == group
                ).sort(by='date')
                plt.plot(
                    plot_df['date'],
                    (plot_df['pct_change'] + 1).cum_prod(),
                    label=group, linewidth=2
                )
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{(y - 1):.0%}')
            )
            if group_df.shape[0] > 20:
                n = max(1, len(group_df) // 10)
                plt.xticks(group_df['date'][::n], rotation=45)
            else:
                plt.xticks(rotation=45)
            plt.title('group return', fontsize=16)
            plt.xlabel('date', fontsize=12)
            plt.ylabel('return', fontsize=12)
            plt.tight_layout()
            plt.show()
        if return_df:  # 返回DataFrame
            return group_df
        return None


