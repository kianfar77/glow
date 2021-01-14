# Copyright 2019 The Glow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .ridge_udfs import *
from nptyping import Float, NDArray
import pandas as pd
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pyspark.sql.functions as f
from typeguard import typechecked
from typing import Any, Dict, List, Union
from glow.logging import record_hls_event
import warnings

# Ignore warning to use applyInPandas instead of apply
# TODO(hhd): Remove this and start using applyInPandas once we only support Spark 3.x.
warnings.filterwarnings('ignore', category=UserWarning, message='.*applyInPandas.*')

__all__ = ['RidgeReduction']


@typechecked
class RidgeReduction:
    """
    The RidgeReducer class is intended to reduce the feature space of an N by M block matrix X to an N by P<<M block
    matrix.  This is done by fitting K ridge models within each block of X on one or more target labels, such that a
    block with L columns to begin with will be reduced to a block with K columns, where each column is the prediction
    of one ridge model for one target label.
    """
    def __init__(self,
                 block_df: DataFrame,
                 label_df: pd.DataFrame,
                 sample_blocks: Dict[str, List[str]],
                 cov_df: pd.DataFrame = pd.DataFrame({}),
                 alphas: NDArray[(Any, ), Float] = np.array([]),
                 label_type='detect') -> None:
        """
        Args:
            block_df : Spark DataFrame representing the beginning block matrix X
            label_df : Pandas DataFrame containing the target labels used in fitting the ridge models
            sample_blocks : Dict containing a mapping of sample_block ID to a list of corresponding sample IDs
            cov_df : Pandas DataFrame containing covariates to be included in every model in the stacking
                ensemble (optional).
            alphas : array_like of alpha values used in the ridge reduction (optional).
            label_type:
        """
        self.set_block_df(block_df)
        self.set_sample_blocks(sample_blocks)
        self.set_label_type(label_type)
        self.set_label_df(label_df)
        self.set_cov_df(cov_df)
        self.set_alphas(alphas)
        self.set_model_df()
        self.set_reduced_block_df()

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_RidgeReduction__block_df'], state['_RidgeReduction__model_df'], state[
            '_RidgeReduction__reduced_block_df']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.__block_df = [x for x in globals().values() if id(x) == self.__block_df_id]
        self.__model_df = [x for x in globals().values() if id(x) == self.__model_df_id]
        self.__reduced_block_df = [
            x for x in globals().values() if id(x) == self.__reduced_block_df_id
        ]

    def set_block_df(self, block_df: DataFrame) -> None:
        self.__block_df = block_df
        self.__block_df_id = id(self.__block_df)

    def set_sample_blocks(self, sample_blocks: Dict[str, List[str]]) -> None:
        self.__sample_blocks = sample_blocks

    def set_label_type(self, label_type: str) -> None:
        self.__label_type = label_type

    def set_label_df(self, label_df: pd.DataFrame) -> None:
        self.__label_df = label_df
        self.__is_binary = is_binary(label_df)
        self.__std_label_df = prepare_labels_and_warn(label_df, self.__is_binary, self.__label_type)

    def set_cov_df(self, cov_df: pd.DataFrame) -> None:
        self.__cov_df = cov_df
        self.__std_cov_df = prepare_covariates(cov_df)

    def set_alphas(self, alphas: NDArray[(Any, ), Float]) -> None:
        self.__alphas = generate_alphas(
            self.__block_df) if alphas.size == 0 else create_alpha_dict(alphas)

    def set_model_df(self, model_df: DataFrame = None) -> None:
        self.__model_df = model_df
        self.__model_df_id = id(self.__model_df)

    def set_reduced_block_df(self, reduced_block: DataFrame = None) -> None:
        self.__reduced_block_df = reduced_block
        self.__reduced_block_df_id = id(self.__reduced_block_df)

    def is_binary(self) -> bool:
        return self.__is_binary

    def get_block_df(self) -> DataFrame:
        return self.__block_df

    def get_sample_blocks(self) -> Dict[str, List[str]]:
        return self.__sample_blocks

    def get_label_type(self) -> str:
        return self.__label_type

    def get_label_df(self) -> pd.DataFrame:
        return self.__label_df

    def get_std_label_df(self) -> pd.DataFrame:
        return self.__std_label_df

    def get_cov_df(self) -> pd.DataFrame:
        return self.__cov_df

    def get_std_cov_df(self) -> pd.DataFrame:
        return self.__std_cov_df

    def get_cov_df(self) -> pd.DataFrame:
        return self.__cov_df

    def get_alphas(self) -> Dict[str, Float]:
        return self.__alphas

    def get_model_df(self) -> DataFrame:
        return self.__model_df

    def get_reduced_block_df(self) -> DataFrame:
        return self.__reduced_block_df

    def fit(self) -> DataFrame:
        """
        Fits a ridge reducer model, represented by a Spark DataFrame containing coefficients for each of the ridge
        alpha parameters, for each block in the starting matrix, for each label in the target labels.

        Returns:
            Spark DataFrame containing the model resulting from the fitting routine.
        """

        map_key_pattern = ['header_block', 'sample_block']
        reduce_key_pattern = ['header_block', 'header']

        if 'label' in self.__block_df.columns:
            map_key_pattern.append('label')
            reduce_key_pattern.append('label')

        map_udf = pandas_udf(
            lambda key, pdf: map_normal_eqn(key, map_key_pattern, pdf, self.__std_label_df, self.
                                            __sample_blocks, self.__std_cov_df), normal_eqn_struct,
            PandasUDFType.GROUPED_MAP)
        reduce_udf = pandas_udf(lambda key, pdf: reduce_normal_eqn(key, reduce_key_pattern, pdf),
                                normal_eqn_struct, PandasUDFType.GROUPED_MAP)
        model_udf = pandas_udf(
            lambda key, pdf: solve_normal_eqn(key, map_key_pattern, pdf, self.__std_label_df, self.
                                              __alphas, self.__std_cov_df), model_struct,
            PandasUDFType.GROUPED_MAP)

        record_hls_event('wgrRidgeReduceFit')

        self.set_model_df(
            self.__block_df.groupBy(map_key_pattern).apply(map_udf).groupBy(
                reduce_key_pattern).apply(reduce_udf).groupBy(map_key_pattern).apply(model_udf))

        return self.__model_df

    def transform(self) -> DataFrame:
        """
        Transforms a starting block matrix to the reduced block matrix, using a reducer model produced by the
        RidgeReducer fit method.

        Returns:
             Spark DataFrame representing the reduced block matrix
        """
        check_model(self.__model_df)

        transform_key_pattern = ['header_block', 'sample_block']

        if 'label' in self.__block_df.columns:
            transform_key_pattern.append('label')
            joined = self.__block_df.drop('sort_key') \
                .join(self.__model_df, ['header_block', 'sample_block', 'header'], 'right') \
                .withColumn('label', f.coalesce(f.col('label'), f.col('labels').getItem(0)))
        else:
            joined = self.__block_df.drop('sort_key') \
                .join(self.__model_df, ['header_block', 'sample_block', 'header'], 'right')

        transform_udf = pandas_udf(
            lambda key, pdf: apply_model(key, transform_key_pattern, pdf, self.__std_label_df, self.
                                         __sample_blocks, self.__alphas, self.__std_cov_df),
            reduced_matrix_struct, PandasUDFType.GROUPED_MAP)

        record_hls_event('wgrRidgeReduceTransform')

        self.set_reduced_block_df(joined.groupBy(transform_key_pattern).apply(transform_udf))

        return self.__reduced_block_df

    def fit_transform(self) -> DataFrame:
        """
        Fits a ridge reducer model with a block matrix, then transforms the matrix using the model.

        Returns:
            Spark DataFrame representing the reduced block matrix
        """

        self.fit()
        return self.transform()
