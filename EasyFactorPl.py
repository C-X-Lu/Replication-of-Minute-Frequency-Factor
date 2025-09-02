import polars as pl
import os
from typing import Optional, List, Literal
from joblib import Parallel, delayed
from tqdm import tqdm
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
    def _read_exposure(
            factor_name: str, path: Optional[str|None], default_path: str
    ) -> Optional[pl.DataFrame|None]:
        """
        读取因子暴露数据
        :param factor_name: 因子名
        :param path: 因子保存的路径，可以为文件或文件夹
        :param default_path: 默认路径，在path为空时使用
        :return:
        """
        factor_exposure = None
        if path is None:
            path = default_path
        if path.endswith('.parquet'):  # 传递的path参数为因子暴露所在的位置
            factor_exposure = pl.read_parquet(path)
        else:  # 传递的path参数为文件夹
            exposures = os.listdir(path)
            if f'{factor_name}.parquet' in exposures:  # 存在已计算的因子暴露
                path = os.path.join(path, f'{factor_name}.parquet')
                factor_exposure = pl.read_parquet(path)
        return factor_exposure

    @staticmethod
    def _process_single_file(file_name, folder_path, calculate_method, pv_data):
        """处理单个文件"""
        try:
            file_path = os.path.join(folder_path, file_name)
            if pv_data is None:  # 不需要日频量价数据
                return calculate_method(pl.read_parquet(file_path))
            else:  # 需要日频量价数据
                return calculate_method(pl.read_parquet(file_path), pv_data)
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            return None

    def cal_exposure_by_min_data(
            self,
            calculate_method,
            path: str=None,
            need_daily_data: bool=False,
            n_jobs: int=None
    ):
        r"""
        使用分钟频数据计算因子暴露。如果已有已计算的部分则更新至最新数据。
        :param calculate_method: 因子计算方法
        :param path: 因子暴露的保存路径，默认为"\QuantData\MinuteFreqFactor"
        :param need_daily_data: 是否需要日频量价数据，默认为False
        :param n_jobs:
        """
        factor_exposure = self._read_exposure(
            factor_name=self.factor_name,
            default_path=r'\QuantData\MinuteFreqFactor',
            path=path
        )

        folder_path = r'D:\QuantData\KLine_cleaned'  # 分钟频价量数据
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        pv_data_index = pl.DataFrame(
            data={'file_name': file_names},
            schema={'file_name': pl.String},
        ).with_columns(
            pl.col('file_name')
            .str.head(8)
            .str.to_date(format='%Y%m%d')
            .alias('date')
        )
        if factor_exposure is not None:  # 如果有已计算的因子暴露
            end_date = factor_exposure['date'].max()
            pv_data_index = pv_data_index.filter(pl.col('date') > end_date)

        valid_results = []
        if len(pv_data_index) > 0:
            if need_daily_data & (n_jobs is None):  # 如果需要日频量价数据
                daily_pv = pl.read_parquet(r'\QuantData\Price_volume.parquet')
                n_jobs = 1
            elif need_daily_data & (n_jobs is not None):
                daily_pv = pl.read_parquet(r'\QuantData\Price_volume.parquet')
                n_jobs = n_jobs
            elif (~need_daily_data) & (n_jobs is None):
                daily_pv = None
                n_jobs = -1
            elif (~need_daily_data) & (n_jobs is not None):
                daily_pv = None
                n_jobs = n_jobs
            else:
                raise ValueError(
                    f'不正确的need_daily_data或n_jobs参数值, {need_daily_data}与{n_jobs}｝'
                )
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._process_single_file)(
                    file_name,
                    folder_path,
                    calculate_method,
                    daily_pv
                )
                for file_name in tqdm(pv_data_index['file_name'], desc='Processing')
            )
            valid_results = [r for r in results if r is not None]

        if factor_exposure is None:
            self.factor_exposure = (
                pl.concat(valid_results, how='vertical')
                .sort(['date', 'code'])
            )
        elif len(valid_results) > 0:
            update_exposure = pl.concat(valid_results, how='vertical')
            self.factor_exposure = (
                pl.concat(
                    items=[factor_exposure, update_exposure],
                    how='vertical'
                )
                .sort(['date', 'code'])
            )
        else:
            self.factor_exposure = factor_exposure

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
            pl.scan_parquet(r'\QuantData\Price_Volume.parquet')
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

    def cal_exposure_by_intermediate(
            self,
            calculate_method,
            intermediate: Optional[str|List[str]],
            path: str=None,
            intermediate_path: str=None,
            depend_past_days: int=20,
            need_daily_pv_data: bool=False,
            depend_columns: List[str]=None
    ):
        r"""
        使用中间因子计算因子暴露。
        :param calculate_method: 计算方法
        :param intermediate: 中间变量。可以传递为"PanDeng"或["ZhaoMoChenWu", "WuBiGuMu"]
        :param path: 因子暴露的保存路径，默认为"\QuantData\MinuteFreqFactor’
        :param intermediate_path: 中间因子的保存路径，默认为‘\QuantData\MinuteFreqFactor\FangZheng Factor\Intermediate Factor’
        :param depend_past_days: 依赖过去多少天的值，默认为20
        :param need_daily_pv_data: 是否需要日频量价数据
        :param depend_columns: 依赖的列，在需要日频量价数据时可以传递
        """
        factor_exposure = self._read_exposure(
            factor_name=self.factor_name,
            default_path=r'\QuantData\MinuteFreqFactor',
            path=path
        )

        minute_freq_factors = [
            file_name.removesuffix('.parquet')
            for file_name in os.listdir(r'\QuantData\MinuteFreqFactor')
        ]  # 分钟频因子集
        daily_freq_factors = [
            file_name.removesuffix('.parquet')
            for file_name in os.listdir(r'\QuantData\DailyFreqFactor')
        ]  # 日频因子集
        if intermediate_path is None:
            special_factors = []
        else:
            special_factors = [
                file_name.removesuffix('.parquet')
                for file_name in os.listdir(intermediate_path)
            ]  # 指定因子集

        def _get_intermediate_path(intermediate_name: str, folder_path: str=None) -> str:
            """获得中间变量的路径"""
            if intermediate_name in minute_freq_factors:
                if folder_path is None:
                    folder_path = (
                        r'\QuantData\MinuteFreqFactor'
                    )
                return os.path.join(
                    folder_path,
                    f'{intermediate_name}.parquet'
                )
            elif intermediate_name in daily_freq_factors:
                if folder_path is None:
                    folder_path = (
                        r'\QuantData\DailyFreqFactor'
                    )
                return os.path.join(
                    folder_path,
                    f'{intermediate_name}.parquet'
                )
            elif intermediate_name in special_factors:
                if folder_path is None:
                    folder_path = intermediate_path
                return os.path.join(
                    folder_path,
                    f'{intermediate_name}.parquet'
                )
            else:
                raise ValueError(f'中间变量{intermediate_name}未计算')

        if isinstance(intermediate, str):  # 中间变量仅为一个因子
            intermediate_df = pl.read_parquet(_get_intermediate_path(
                intermediate, intermediate_path
            ))
        elif isinstance(intermediate, list):  # 中间变量有多个因子
            intermediate_df = []
            for single_intermediate in intermediate:
                intermediate_df.append(
                    pl.read_parquet(_get_intermediate_path(
                        single_intermediate, intermediate_path
                    ))
                )
            intermediate_df = pl.concat(intermediate_df, how='align')
        else:
            raise ValueError(f'中间变量{intermediate}格式不正确')

        if need_daily_pv_data:  # 如果需要日频量价数据
            intermediate_df = pl.concat(
                [
                    intermediate_df,
                    self._read_daily_pv_data(depend_columns)
                ], how='align'
            )

        if factor_exposure is None:  # 不存在已计算的因子暴露
            factor_exposure = calculate_method(intermediate_df)
        else:  # 存在已计算的因子暴露
            if depend_past_days is None:
                latest_date = factor_exposure['date'].max()
            else:
                latest_date = factor_exposure.select(
                    pl.col('date')
                    .unique()
                    .top_k(depend_past_days + 1)
                )['date'].min()
            intermediate_df = intermediate_df.filter(
                pl.col('date') > latest_date
            )
            newly_factor_exposure = calculate_method(intermediate_df)
            if newly_factor_exposure is not None:
                factor_exposure = (
                    pl.concat([factor_exposure, ])
                    .unique(subset=['code', 'date'])
                )

        self.factor_exposure = factor_exposure

    def cal_final_exposure(
            self,
            frequency: str|int,
            method:str,
            mode: str='calendar',
            pool='full'
    ) -> pl.DataFrame:
        """
        计算最终因子暴露。
        股票池暂不支持。
        :param frequency: 频率：周频'weekly'、月频'monthly'或 t 日频
        :param method: 计算方法：'o' 取最后一个有效值、'm' 取算数平均、'z' 取Z-score分、'std' 取当期标准差
        :param mode: 重采样模式：calendar按日历重采样，days按天数重采样。默认为calendar
        :param pool: 股票池：'full'全市场、'300'沪深300成分股、'500'中证500成分股、'1000'中证1000成分股
        :return: pd.DataFrame: 最终的因子暴露
        """
        if mode == 'calendar':
            if frequency == 'weekly':
                group_param = '1w'
            elif frequency == 'monthly':
                group_param = '1mo'
            else:
                raise ValueError(f'Unsupported frequency for calendar: {frequency}')
            if pool == 'full':
                pass
            else:
                raise ValueError(f'不支持的股票池: {pool}')
            name = f'{frequency}_{self.factor_name}_{method}'
            if method == 'o':
                final_exposure = (
                    self.factor_exposure
                    .group_by_dynamic(every=group_param, group_by='code')
                    .agg(
                        pl.col(self.factor_name)
                        .last()
                        .alias(name)
                    )
                )
            elif method == 'm':
                final_exposure = (
                    self.factor_exposure
                    .group_by_dynamic(every=group_param, group_by='code')
                    .agg(
                        pl.col(self.factor_name)
                        .mean()
                        .alias(name)
                    )
                )
            elif method == 'z':
                final_exposure = (
                    self.factor_exposure
                    .group_by_dynamic(every=group_param, group_by='code')
                    .agg(
                        (
                            (
                                pl.col(self.factor_name).last()
                                - pl.col(self.factor_name).mean()
                            ) / pl.col(self.factor_name).std()
                        ).alias(name)
                    )
                )
            elif method == 'std':
                final_exposure = (
                    self.factor_exposure
                    .group_by_dynamic(every=group_param, group_by='code')
                    .agg(
                        pl.col(self.factor_name)
                        .std()
                        .alias(name)
                    )
                )
            else:
                raise ValueError('Unknown method')
        elif mode == 'days':
            if isinstance(frequency, int):
                name = f'{self.factor_name}_{frequency}_{method}'
                if method == 'o':
                    final_exposure = (
                        self.factor_exposure.select(
                            pl.col('code'),
                            pl.col('date'),
                            pl.col(self.factor_name)
                            .alias(name)
                        )
                    )
                elif method == 'm':
                    final_exposure = (
                        self.factor_exposure.select(
                            pl.col('code'),
                            pl.col('date'),
                            pl.col(self.factor_name)
                            .rolling_mean(frequency, min_samples=frequency)
                            .over('code')
                            .alias(name)
                        )
                    )
                elif method == 'z':
                    final_exposure = (
                        self.factor_exposure.select(
                            pl.col('code'),
                            pl.col('date'),
                            (
                                (
                                    pl.col(self.factor_name)
                                    - pl.col(self.factor_name)
                                    .rolling_mean(frequency, min_samples=frequency)
                                ) / (
                                    pl.col(self.factor_name)
                                    .rolling_std(frequency, min_samples=frequency, ddof=0)
                                )
                            ).over('code')
                            .alias(name)
                        )
                    )
                elif method == 'std':
                    final_exposure = (
                        self.factor_exposure.select(
                            pl.col('code'),
                            pl.col('date'),
                            pl.col(self.factor_name)
                            .rolling_std(frequency, min_samples=frequency, ddof=0)
                            .over('code')
                            .alias(name)
                        )
                    )
                else:
                    raise ValueError('Unknown method')
            else:
                raise ValueError(f'Unsupported frequency for days: {frequency}')
        else:
            raise ValueError(f'Unknown mode: {mode}')
        return final_exposure

    def to_parquet(self, path: str=None):
        r"""
        将数据保存为parquet。
        :param path: 保存的路径, 默认路径为'\QuantData\MinuteFreqFactor', 默认名称为因子名。
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

