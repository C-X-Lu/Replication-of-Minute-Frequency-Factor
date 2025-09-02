from Factor import Factor
import os
import polars as pl
from typing import Optional
from joblib import Parallel, delayed
from tqdm import tqdm

class MinFreqFactor(Factor):
    def __init__(self, factor_name, factor_exposure=None):
        """
        分钟频因子类：从因子类中继承coverage/ic_test/group_test
        :param factor_name: 因子的名字
        :param factor_exposure: 因子暴露
        """
        super().__init__(factor_name, factor_exposure)

    @staticmethod
    def _process_single_file(file_name, folder_path, calculate_method):
        """处理单个文件"""
        try:
            file_path = os.path.join(folder_path, file_name)
            return calculate_method(pl.read_parquet(file_path))
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            return None

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

    def cal_exposure_by_min_data(
            self,
            calculate_method,
            path: str = None,
            n_jobs: int = None
    ):
        r"""
        使用分钟频数据计算因子暴露。如果已有已计算的部分则更新至最新数据。
        :param calculate_method: 因子计算方法
        :param path: 因子暴露的保存路径，默认为‘D:\quant\MinuteFreqFactor’
        :param n_jobs:
        """
        factor_exposure = self._read_exposure(
            factor_name=self.factor_name,
            default_path=r'D:\QuantData\MinuteFreqFactor\CICC Factor',
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
            if n_jobs is None:  # 如果需要日频量价数据
                n_jobs = -1
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._process_single_file)(
                    file_name,
                    folder_path,
                    calculate_method
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