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

import glow
from glow import *
from glow.wgr.functions import *
from glow.wgr.linear_model import *

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd


def test_two_contigs(spark):
    # spark \
    #     .read \
    #     .format("plink") \
    #     .option('famDelimiter', '\t') \
    #     .load("test-data/regenie/example.bed") \
    #     .withColumn('referenceAllele', lit('C')) \
    #     .withColumn('alternateAlleles', array(lit('A'))) \
    #     .withColumn('contigName',
    #           when(col('start') > 499, '2').otherwise('1')
    #           ).write.format('bigbgen').save('test-data/regenie/myexample.bgen')



    spark \
        .read \
        .format("bgen") \
        .load("test-data/regenie/example.bgen") \
        .withColumn('referenceAllele', lit('C')) \
        .withColumn('alternateAlleles', array(lit('A'))) \
        .withColumn('contigName',
                    when(col('start') > 499, '2').otherwise('1')
                    ).write.format('bigbgen').save('test-data/regenie/myexample.bgen')

    # df = spark \
    #     .read \
    #     .format("bgen") \
    #     .load("test-data/regenie/myexample.bgen") \
    #
    # df.show(1000)

    # df.write.format('bigbgen').save('test-data/regenie/myexample.bgen')

def test_regenie_example(spark):
    base_variant_df = (spark \
        .read \
        .format("bgen") \
        .load("test-data/regenie/example.bgen") \
        .withColumn('referenceAllele', lit('C')) \
        .withColumn('alternateAlleles', array(lit('A'))) \
        # Move the second half to chr2
        .withColumn('contigName', when(col('start') > 499, '2').otherwise('1')) \
        # Make hard calls on bgen
        .withColumn('genotypes', expr("transform(genotypes, g -> add_struct_fields(subset_struct(g, 'sampleId', 'phased', 'ploidy'), 'calls', hard_calls(g.posteriorProbabilities, 1, False)))")) \
        .cache()
        )

    # base_variant_df.show(truncate=False)


    # base_variant_df = spark \
    #     .read \
    #     .format("plink") \
    #     .option('famDelimiter', '\t') \
    #     .load("test-data/regenie/example.bed") \
    #     .withColumn('referenceAllele', lit('C')) \
    #     .withColumn('alternateAlleles', array(lit('A')))
    # .withColumn('contigName',
    #           when(col('start') > 499, '2').otherwise('1')
    #           ).write.format('bigbgen').save('test-data/regenie/myexample.bgen')

    # base_variant_df.show(1, truncate=False)
    # # Split regenie example onto 2 contigs
    # base_variant_df.printSchema()
    # base_variant_df.withColumn('contigName',
    #                           when(col('start') > 499, '2').otherwise('1')
    #                           ).write.format('bigbgen').save('test-data/regenie/myexample.bgen')

    # from pdb_clone import pdb
    # pdb.set_trace_remote()

    # .transform('split_multiallelics', base_variant_df) \

    variant_df = (base_variant_df \
        .withColumn('values', mean_substitute(genotype_states(col('genotypes')))) \
        # Order of alleles in our bgen reader is the opposit of regenie bgen reader
        # To have the same values as in regenie with myexample.bgen (created in test_two_contigs) we complement the values
        .withColumn('values', expr("transform(values, v -> 2-v)")) \
        .filter(size(array_distinct('values')) > 1) \
        .alias('variant_df')
        )

    # variant_df.show(1, truncate=False)

    # variant_df.write.format('bigbgen').save('test-data/regenie/myexample.bgen')
    # """
    label_df = pd.read_csv('test-data/regenie/phenotype_bin.txt', sep='\s')
    label_df['sample_id'] = label_df['FID'].astype(str) + '_' + label_df['IID'].astype(str)
    label_df = label_df.drop(columns=['FID', 'IID']).set_index('sample_id')
    label_df_mean_centered = (label_df - label_df.mean())

    # covariate_df = pd.read_csv('test-data/regenie/covariates.txt', sep='\s')
    # covariate_df['sample_id'] = covariate_df['FID'].astype(str) + '_' + covariate_df['IID'].astype(str)
    # covariate_df = covariate_df.drop(columns=['FID', 'IID']).set_index('sample_id')
    # covariate_df = (covariate_df - covariate_df.mean()) / covariate_df.std()
    # """
    sample_ids = get_sample_ids(variant_df)
    # print(covariate_df.mean(0))
    #
    # print(label_df)
    # print(covariate_df)
    variants_per_block = 100

    sample_block_count = 5

    block_df, sample_blocks = block_variants_and_samples(variant_df,
                                                         sample_ids,
                                                         variants_per_block,
                                                         sample_block_count)

    # block_df.show()
    # print(sample_blocks)
    # """
    stack = RidgeReducer()
    model_df = stack.fit(block_df, label_df_mean_centered, sample_blocks)
    # model_df.orderBy('sample_block', 'sort_key').show(truncate=False)
    # model_df.toPandas().to_csv('test-data/regenie/model_df.csv', sep="\t")
    model_df.filter('sample_block = "1" and header_block="chr_1_block_0"') \
        .withColumn('alphas', slice('alphas', 9, 10)) \
        .withColumn('labels', slice('labels', 9, 10)) \
        .withColumn('coefficients', slice('coefficients', 9, 10))
        # .show(200, truncate=False)

    model_array = model_df.filter('sample_block = "1" and header_block="chr_1_block_0"') \
        .select(slice('coefficients', 9, 10))

    # model_array.toPandas()

    reduced_block_df = stack.transform(block_df,
                                       label_df_mean_centered,
                                       sample_blocks,
                                       model_df)

    reduced_block_df.filter('alpha="alpha_4" and label="Y1" and header like "%chr_1_block_0%"').orderBy('sample_block').show(truncate=False)


    estimator = LogisticRegression()
    model_df, cv_df = estimator.fit(reduced_block_df,
                                    label_df,
                                    sample_blocks)
                                    # covariate_df)

    model_df.printSchema()
    cv_df.printSchema()
    model_df.show(truncate=False)
    cv_df.show(truncate=False)



    y_hat_df = estimator.transform_loco(
        reduced_block_df,
        label_df,
        sample_blocks,
        model_df,
        cv_df)

    # y_hat_df

    y_hat_df.to_csv('test-data/regenie/glowgr.csv')


def test_plink(spark):
    base_variant_df = spark \
        .read \
        .format("bgen") \
        .load("test-data/regenie/example.bgen") \
        .withColumn('referenceAllele', lit('C')) \
        .withColumn('alternateAlleles', array(lit('A')))

    base_variant_df.show(1, truncate=False)

    # .withColumn('genotypes', expr(
    # "transform(genotypes, g -> add_struct_fields(subset_struct(g, 'sampleId', 'phased', 'ploidy'), 'calls', hard_calls(g.posteriorProbabilities, 1, False)))"))

    base_variant_df_plink = spark \
        .read \
        .format("plink") \
        .option('famDelimiter', '\t') \
        .option('bimDelimiter', '\t') \
        .load("test-data/regenie/example.bed") \
        .withColumn('referenceAllele', lit('C')) \
        .withColumn('alternateAlleles', array(lit('A')))

    base_variant_df_plink.show()

    base_variant_df_plink.write.format('bigvcf').save('test-data/regenie/plink.vcf')
    #
    # base_variant_df_vcf = spark \
    #     .read \
    #     .format("vcf") \
    #     .load("test-data/regenie/plink.vcf")
    #
    # base_variant_df_vcf.show(1, truncate=False)
