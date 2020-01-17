/*
 * Copyright 2019 The Glow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.projectglow.transformers.normalizevariants

import io.projectglow.Glow
import io.projectglow.common.{CommonOptions, GlowLogging, VCFOptions}
import io.projectglow.sql.GlowBaseTest
import org.apache.spark.SparkConf

class NormalizeVariantsTransformerSuite extends GlowBaseTest with GlowLogging {

  lazy val sourceName: String = "vcf"
  lazy val testFolder: String = s"$testDataHome/variantnormalizer-test"

  // gatk test file (multiallelic)
  // The base of vcfs and reference in these test files were taken from gatk
  // LeftTrimAndLeftAlign test suite. The reference genome was trimmed to +/-400 bases around
  // each variant to generate a small reference fasta. The vcf variants were modified accordingly.

  lazy val gatkTestReference =
    s"$testFolder/Homo_sapiens_assembly38.20.21_altered.fasta"

  lazy val gatkTestVcf =
    s"$testFolder/test_left_align_hg38_altered.vcf"

  lazy val gatkTestVcfExpectedSplit =
    s"$testFolder/test_left_align_hg38_altered_vtdecompose.vcf"

  lazy val gatkTestVcfExpectedNormalized =
    s"$testFolder/test_left_align_hg38_altered_bcftoolsnormalized.vcf"

  lazy val gatkTestVcfExpectedSplitNormalized =
    s"$testFolder/test_left_align_hg38_altered_vtdecompose_bcftoolsnormalized.vcf"

  // These files are similar to above but contain symbolic variants.
  lazy val gatkTestVcfSymbolic =
    s"$testFolder/test_left_align_hg38_altered_symbolic.vcf"

  lazy val gatkTestVcfSymbolicExpectedSplit =
    s"$testFolder/test_left_align_hg38_altered_symbolic_vtdecompose.vcf"

  lazy val gatkTestVcfSymbolicExpectedNormalized =
    s"$testFolder/test_left_align_hg38_altered_symbolic_bcftoolsnormalized.vcf"

  lazy val gatkTestVcfSymbolicExpectedSplitNormalized =
    s"$testFolder/test_left_align_hg38_altered_symbolic_vtdecompose_bcftoolsnormalized.vcf"

  // vt test files
  // The base of vcfs and reference in these test files were taken from vt
  // (https://genome.sph.umich.edu/wiki/Vt) normalization test suite. The vcf in this test suite
  // is biallelic. The reference genome was trimmed to +/-100 bases around each variant to
  // generate a small reference fasta. The vcf variants were modified accordingly.
  //
  // The multialleleic versions were generated by artificially adding more alleles and
  // corresponding genotypes to some of the variants.
  lazy val vtTestReference = s"$testFolder/20_altered.fasta"

  lazy val vtTestVcfBiallelic =
    s"$testFolder/01_IN_altered_biallelic.vcf"

  lazy val vtTestVcfBiallelicExpectedSplit =
    s"$testFolder/01_IN_altered_biallelic_vtdecompose.vcf"

  lazy val vtTestVcfBiallelicExpectedNormalized =
    s"$testFolder/01_IN_altered_biallelic_bcftoolsnormalized.vcf"

  lazy val vtTestVcfBiallelicExpectedSplitNormalized =
    s"$testFolder/01_IN_altered_biallelic_vtdecompose_bcftoolsnormalized.vcf"

  lazy val vtTestVcfMultiAllelic =
    s"$testFolder/01_IN_altered_multiallelic.vcf"

  lazy val vtTestVcfMultiAllelicExpectedSplit =
    s"$testFolder/01_IN_altered_multiallelic_vtdecompose.vcf"

  lazy val vtTestVcfMultiAllelicExpectedNormalized =
    s"$testFolder/01_IN_altered_multiallelic_bcftoolsnormalized.vcf"

  lazy val vtTestVcfMultiAllelicExpectedSplitNormalized =
    s"$testFolder/01_IN_altered_multiallelic_vtdecompose_bcftoolsnormalized.vcf"

  override def sparkConf: SparkConf = {
    super
      .sparkConf
      .set(
        "spark.hadoop.io.compression.codecs",
        "org.seqdoop.hadoop_bam.util.BGZFCodec"
      )
  }

  /**
   *  Tests whether the transformed VCF matches the expected VCF
   */
  def testNormalizedvsExpected(
      originalVCFFileName: String,
      expectedVCFFileName: String,
      referenceGenome: Option[String],
      mode: Option[String],
      includeSampleIds: Boolean
  ): Unit = {

    val options: Map[String, String] = Map() ++ {
        referenceGenome match {
          case Some(r) => Map("referenceGenomePath" -> r)
          case None => Map()
        }
      } ++ {
        mode match {
          case Some(m) => Map("mode" -> m)
          case None => Map()
        }
      }

    val dfOriginal = spark
      .read
      .format(sourceName)
      .options(Map(CommonOptions.INCLUDE_SAMPLE_IDS -> includeSampleIds.toString))
      .load(originalVCFFileName)

    val dfNormalized = Glow
      .transform(
        "normalize_variants",
        dfOriginal,
        options
      )
      .orderBy("contigName", "start", "end")

    val dfExpected = spark
      .read
      .format(sourceName)
      .options(Map(CommonOptions.INCLUDE_SAMPLE_IDS -> includeSampleIds.toString))
      .load(expectedVCFFileName)
      .orderBy("contigName", "start", "end")

    val dfExpectedColumns =
      dfExpected.columns.map(name => if (name.contains(".")) s"`${name}`" else name)

    assert(dfNormalized.count() == dfExpected.count())

    dfExpected
      .drop("splitFromMultiAllelic")
      .collect
      .zip(
        dfNormalized
          .select(dfExpectedColumns.head, dfExpectedColumns.tail: _*) // make order of columns the same
          .drop("splitFromMultiAllelic")
          .collect
      )
      .foreach {
        case (rowExp, rowNorm) =>
          assert(rowExp.equals(rowNorm), s"Expected\n$rowExp\nNormalized\n$rowNorm")
      }
  }

  def testNormalizedvsExpected(
      originalVCFFileName: String,
      expectedVCFFileName: String,
      referenceGenome: Option[String],
      mode: Option[String]
  ): Unit = {
    testNormalizedvsExpected(originalVCFFileName, expectedVCFFileName, referenceGenome, mode, true)
  }

  test("normalization transform do-normalize-no-split no-reference") {
    // vcf containing multi-allelic variants
    try {
      testNormalizedvsExpected(vtTestVcfMultiAllelic, vtTestVcfMultiAllelic, None, None)
    } catch {
      case _: IllegalArgumentException => succeed
      case _: Throwable => fail()
    }
  }

  test("normalization transform do-normalize-no-split") {

    testNormalizedvsExpected(
      vtTestVcfBiallelic,
      vtTestVcfBiallelicExpectedNormalized,
      Option(vtTestReference),
      Option("normalize"))

    testNormalizedvsExpected(
      vtTestVcfMultiAllelic,
      vtTestVcfMultiAllelicExpectedNormalized,
      Option(vtTestReference),
      None)

    // without sampleIds
    testNormalizedvsExpected(
      vtTestVcfMultiAllelic,
      vtTestVcfMultiAllelicExpectedNormalized,
      Option(vtTestReference),
      None,
      false)

    testNormalizedvsExpected(
      gatkTestVcf,
      gatkTestVcfExpectedNormalized,
      Option(gatkTestReference),
      Option("normalize"))

    testNormalizedvsExpected(
      gatkTestVcfSymbolic,
      gatkTestVcfSymbolicExpectedNormalized,
      Option(gatkTestReference),
      None)

  }

  test("normalization transform no-normalize-do-split") {

    testNormalizedvsExpected(
      vtTestVcfBiallelic,
      vtTestVcfBiallelicExpectedSplit,
      None,
      Option("split")
    )

    testNormalizedvsExpected(
      vtTestVcfMultiAllelic,
      vtTestVcfMultiAllelicExpectedSplit,
      None,
      Option("split")
    )

    // without sampleIds
    testNormalizedvsExpected(
      vtTestVcfMultiAllelic,
      vtTestVcfMultiAllelicExpectedSplit,
      None,
      Option("split"),
      false
    )

    testNormalizedvsExpected(
      gatkTestVcf,
      gatkTestVcfExpectedSplit,
      None,
      Option("split")
    )

    testNormalizedvsExpected(
      gatkTestVcfSymbolic,
      gatkTestVcfSymbolicExpectedSplit,
      Option(gatkTestReference),
      Option("split"))

  }

  test("normalization transform do-normalize-do-split") {

    testNormalizedvsExpected(
      vtTestVcfBiallelic,
      vtTestVcfBiallelicExpectedNormalized,
      Option(vtTestReference),
      Option("split_and_normalize"))

    testNormalizedvsExpected(
      vtTestVcfMultiAllelic,
      vtTestVcfMultiAllelicExpectedSplitNormalized,
      Option(vtTestReference),
      Option("split_and_normalize"))

    // without sampleIds
    testNormalizedvsExpected(
      vtTestVcfMultiAllelic,
      vtTestVcfMultiAllelicExpectedSplitNormalized,
      Option(vtTestReference),
      Option("split_and_normalize"),
      false)

    testNormalizedvsExpected(
      gatkTestVcf,
      gatkTestVcfExpectedSplitNormalized,
      Option(gatkTestReference),
      Option("split_and_normalize"))

    testNormalizedvsExpected(
      gatkTestVcfSymbolic,
      gatkTestVcfSymbolicExpectedSplitNormalized,
      Option(gatkTestReference),
      Option("split_and_normalize"))

  }

}
