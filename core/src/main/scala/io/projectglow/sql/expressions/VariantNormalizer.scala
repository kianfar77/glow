package io.projectglow.sql.expressions

import com.google.common.annotations.VisibleForTesting
import htsjdk.samtools.reference.IndexedFastaSequenceFile
import htsjdk.variant.variantcontext._
import io.projectglow.common.GlowLogging
import io.projectglow.common.VariantSchemas._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, LongType, StringType, StructField, StructType}
import org.broadinstitute.hellbender.engine.{ReferenceContext, ReferenceDataSource}
import org.broadinstitute.hellbender.utils.SimpleInterval

import scala.math.min

object VariantNormalizer extends GlowLogging {

  /**
   * Normalizes the input DataFrame of variants and outputs them as a Dataframe
   *
   * @param df                   : Input dataframe of variants
   * @param refGenomePathString  : Path to the underlying reference genome of the variants
   * @return normalized DataFrame
   */
  def normalize(df: DataFrame, refGenomePathString: String): DataFrame = {

    df.select(
      col("*"),
      expr(
        s"""expand_struct(
        |   normalize_variant(
        |       ${contigNameField.name},
        |       ${startField.name},
        |       ${endField.name},
        |       ${refAlleleField.name},
        |       ${alternateAllelesField.name},
        |       lit($refGenomePathString)
        |   )
        |)""".stripMargin)
    )
  }


  /**
   * normalizes a single VariantContext by checking some conditions and then calling realignAlleles
   *
   * @param vc
   * @param refGenomeDataSource
   * @return normalized VariantContext
   */
  def normalizeVariant(
      contigName: String,
      start: Int, // this start is one less that the start used in the alleleblock of the old version of the code.
      end: Int,
      refAllele: String,
      altAlleles: Array[String],
      refGenomeIndexedFasta: IndexedFastaSequenceFile): InternalRow = {
    // validateForNormalization
    var flag = FLAG_UNCHANGED
    var newStart = start
    var newEnd = end
    var newRefAllele = refAllele
    var newAltAlleles = altAlleles

    if (refAllele.isEmpty && altAlleles.isEmpty) {
      // if no alleles, throw exception
      logger.info("REF and ALT alleles are empty allele list...")
      flag = FLAG_ERROR
    } else if (refAllele.length == 1 && altAlleles.forall(_.length == 1)) {
      // if a SNP, do nothing
      flag = FLAG_UNCHANGED
    } else if (altAlleles.isEmpty) {
      // if only one allele and longer than one base, trim to the
      // first base
      newRefAllele = refAllele.take(1)
      flag = FLAG_CHANGED
    } else if (altAlleles.exists(_.matches(".*[<|>|*].*"))) {
      // if any of the alleles is symbolic, do nothing
      flag = FLAG_UNCHANGED
    } else {
      // Create ReferenceDataSource of the reference genome and the AlleleBlock and pass
      // to realignAlleles

      // Trim from right
      var trimSize = 0 // stores total trimSize from right
      var nTrimmedBasesBeforeNextPadding = 0 // stores number of bases trimmed from right before next padding
      var firstAlleleBaseFromRight = refAllele(
        refAllele.length - nTrimmedBasesBeforeNextPadding - 1)

      val allAlleles = refAllele +: altAlleles

      while (allAlleles
          .forall(a => a(a.length - nTrimmedBasesBeforeNextPadding - 1) == firstAlleleBaseFromRight)) {
        // Last base in all alleles are the same

        var padSeq = Array[Byte]()
        var nPadBases = 0

        if (allAlleles
            .map(_.length)
            .min == nTrimmedBasesBeforeNextPadding + 1) {
          // if beginning of any allele is reached, trim from right what
          // needs to be trimmed so far, and pad to the left
          if (newStart > 1) {
            nPadBases = min(PAD_WINDOW_SIZE, newStart - 1)

            refGenomeIndexedFasta.getSubsequenceAt(contigName, start, end)
            // refGenomeDataSource,
            // new SimpleInterval(contig, newStart - 1, newStart - 1)

            refGenomeContext.setWindow(nPadBases - 1, 0)

            padSeq ++= refGenomeContext.getBases()

          } else {
            nTrimmedBasesBeforeNextPadding -= 1
          }

          alleles = alleles.map { a =>
            Allele.create(
              padSeq ++ a
                .getBaseString()
                .dropRight(nTrimmedBasesBeforeNextPadding + 1)
                .getBytes(),
              a.isReference
            )
          }

          trimSize += nTrimmedBasesBeforeNextPadding + 1

          newStart -= nPadBases

          nTrimmedBasesBeforeNextPadding = 0

        } else {

          nTrimmedBasesBeforeNextPadding += 1

        }

        firstAlleleBaseFromRight = alleles(0).getBases()(
          alleles(0).length
          - nTrimmedBasesBeforeNextPadding - 1
        )
      }

      // trim from left
      var nLeftTrimBases = 0
      var firstAlleleBaseFromLeft = alleles(0).getBases()(nLeftTrimBases)
      val minAlleleLength = alleles.map(_.length).min

      while (nLeftTrimBases < minAlleleLength - nTrimmedBasesBeforeNextPadding - 1
        && alleles.forall(_.getBases()(nLeftTrimBases) == firstAlleleBaseFromLeft)) {

        nLeftTrimBases += 1

        firstAlleleBaseFromLeft = alleles(0).getBases()(nLeftTrimBases)
      }

      alleles = alleles.map { a =>
        Allele.create(
          a.getBaseString()
            .drop(nLeftTrimBases)
            .dropRight(nTrimmedBasesBeforeNextPadding)
            .getBytes(),
          a.isReference
        )

      }

      trimSize += nTrimmedBasesBeforeNextPadding

      AlleleBlock(
        alleles,
        newStart + nLeftTrimBases,
        unalignedAlleleBlock.end - trimSize
      )

    }

    InternalRow(newStart, newEnd, newRefAllele, newAltAlleles, flag)

  }
/*
  /**
   * Contains the main normalization logic. Normalizes an AlleleBlock by left aligning and
   * trimming its alleles and adjusting its new start and end.
   *
   * The algorithm has a logic similar to bcftools:
   *
   * It starts from the rightmost base of all alleles and scans one base at a time incrementing
   * trimSize and nTrimmedBasesBeforeNextPadding as long as the bases of all alleles at that
   * position are the same. If the beginning of any of the alleles is reached, all alleles are
   * padded on the left by PAD_WINDOW_SIZE bases by reading from the reference genome amd
   * nTrimmedBaseBeforeNextPadding is reset. The process continues until a position is reached
   * where all alleles do not have the same base or the beginning of the contig is reached. Next
   * trimming from left starts and all bases common among all alleles from left are trimmed.
   * Start and end of the AllleleBlock are adjusted accordingly during the process.
   *
   * @param unalignedAlleleBlock
   * @param refGenomeDataSource
   * @param contigName : contig of the AlleleBlock
   * @return normalized AlleleBlock
   */
  @VisibleForTesting
  private[normalizevariants] def realignAlleles(
      unalignedAlleleBlock: NormalizationResult,
      refGenomeDataSource: ReferenceDataSource,
      contigName: String): NormalizationResult = {

    // Trim from right
    var trimSize = 0 // stores total trimSize from right
    var nTrimmedBasesBeforeNextPadding = 0 // stores number of bases trimmed from right before
    // next padding
    var newStart = unalignedAlleleBlock.start
    var alleles = unalignedAlleleBlock.alleles
    var firstAlleleBaseFromRight = alleles(0).getBases()(
      alleles(0).length
      - nTrimmedBasesBeforeNextPadding - 1
    )

    while (alleles.forall(
        a =>
          a.getBases()(a.length() - nTrimmedBasesBeforeNextPadding - 1) ==
          firstAlleleBaseFromRight
      )) {
      // Last base in all alleles are the same

      var padSeq = Array[Byte]()
      var nPadBases = 0

      if (alleles
          .map(_.length)
          .min == nTrimmedBasesBeforeNextPadding + 1) {
        // if
        // beginning of any allele is reached, trim from right what
        // needs to be trimmed so far, and pad to the left
        if (newStart > 1) {
          nPadBases = min(PAD_WINDOW_SIZE, newStart - 1)

          val refGenomeContext = new ReferenceContext(
            refGenomeDataSource,
            new SimpleInterval(contigName, newStart - 1, newStart - 1)
          )

          refGenomeContext.setWindow(nPadBases - 1, 0)

          padSeq ++= refGenomeContext.getBases()

        } else {
          nTrimmedBasesBeforeNextPadding -= 1
        }

        alleles = alleles.map { a =>
          Allele.create(
            padSeq ++ a
              .getBaseString()
              .dropRight(nTrimmedBasesBeforeNextPadding + 1)
              .getBytes(),
            a.isReference
          )
        }

        trimSize += nTrimmedBasesBeforeNextPadding + 1

        newStart -= nPadBases

        nTrimmedBasesBeforeNextPadding = 0

      } else {

        nTrimmedBasesBeforeNextPadding += 1

      }

      firstAlleleBaseFromRight = alleles(0).getBases()(
        alleles(0).length
        - nTrimmedBasesBeforeNextPadding - 1
      )
    }

    // trim from left
    var nLeftTrimBases = 0
    var firstAlleleBaseFromLeft = alleles(0).getBases()(nLeftTrimBases)
    val minAlleleLength = alleles.map(_.length).min

    while (nLeftTrimBases < minAlleleLength - nTrimmedBasesBeforeNextPadding - 1
      && alleles.forall(_.getBases()(nLeftTrimBases) == firstAlleleBaseFromLeft)) {

      nLeftTrimBases += 1

      firstAlleleBaseFromLeft = alleles(0).getBases()(nLeftTrimBases)
    }

    alleles = alleles.map { a =>
      Allele.create(
        a.getBaseString()
          .drop(nLeftTrimBases)
          .dropRight(nTrimmedBasesBeforeNextPadding)
          .getBytes(),
        a.isReference
      )

    }

    trimSize += nTrimmedBasesBeforeNextPadding

    AlleleBlock(
      alleles,
      newStart + nLeftTrimBases,
      unalignedAlleleBlock.end - trimSize
    )

  }
*/
  private val PAD_WINDOW_SIZE = 100

  private val FLAG_ERROR = "Error"
  private val FLAG_CHANGED = "Changed"
  private val FLAG_UNCHANGED = "Unchanged"

  val normalizedStartField = StructField("normalizedStart", LongType)
  val normalizedEndField = StructField("normalizedEnd", LongType)
  val normalizedRefAlleleField = StructField("normalizedReferenceAllele", StringType)
  val normalizedAlternateAllelesField = StructField("normalizedAlternateAlleles", ArrayType(StringType))
  val normalizationFlagField = StructField("normalizationFlag", StringType)

  val normalizationSchema = StructType(
    Seq(
      normalizedStartField,
      normalizedEndField,
      normalizedRefAlleleField,
      normalizedAlternateAllelesField,
      normalizationFlagField
    )
  )
}
