===============================
Splitting Multiallelic Variants
===============================

.. invisible-code-block: python

    import glow
    glow.register(spark)

    test_dir = 'test-data/variantsplitternormalizer-test/'
    df_original = spark.read.format('vcf').load(test_dir + 'test_left_align_hg38_altered.vcf')
    ref_genome_path = test_dir + 'Homo_sapiens_assembly38.20.21_altered.fasta'

**Splitting multiallelic variants to biallelic variants** is a transformation sometimes required before further downstream analysis. Glow provides the ``split_multiallelics" transformer to be appied on a varaint DataFrame to split multiallelic variants in the DataFrame to biallelic variants.

.. note::

    The splitting logic is the same as the one used by `vt decompose -s <https://genome.sph.umich.edu/wiki/Vt#Decompose>`_ of the vt package. The precise behavior of the splitter is as follows:

    - A given multiallelic row with :math:`n` ``ALT`` alleles is split to :math:`n` biallelic rows, each with one of the ``ALT`` alleles in the alternate alleles column. The ``REF`` allele in all these rows is the same as the ``REF`` allele in the multiallelic row.

    - Each ``INFO`` field is appropriately split among split rows if it has the same number of elements as number of ``ALT`` alleles, otherwise it is repeated in all split rows. A new ``INFO`` column called ``OLD_MULTIALLELIC`` is added to the DataFrame, which for each split row, holds the ``CHROM:POS:REF/ALT`` of its corresponding multiallelic row.

    - Genotype fields are treated as follows: The ``GT`` field becomes biallelic in each row, where ``ALT`` alleles not present in that row are replaced with no call (``.``). The fields with number of entries equal to number of (``REF`` + ``ALT``) alleles, are properly split into rows, where in each split row, only entries corresponding to the ``REF`` allele as well as the ``ALT`` alleles present in that row are kept. The fields which follow colex order (e.g., ``GL``, ``PL``, and ``GP``) are properly split between split rows where in each row only the elements corresponding to genotypes comprising of the ``REF`` and ``ALT`` alleles in that row are listed. Other fields are just repeated over the split rows.

    As an example (shown in VCF file format), the following multiallelic row

    .. code-block::

        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1
        20	101	.	A	ACCA,TCGG	.	PASS	VC=INDEL;AC=3,2;AF=0.375,0.25;AN=8	GT:AD:DP:GQ:PL	0/1:2,15,31:30:99:2407,0,533,697,822,574

    will be split into the following two biallelic rows:

    .. code-block::

        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1
        20	101	.	A	ACCA	.	PASS	VC=INDEL;AC=3;AF=0.375;AN=8;OLD_MULTIALLELIC=20:101:A/ACCA/TCGG	GT:AD:DP:GQ:PL	0/1:2,15:30:99:2407,0,533
        20	101	.	A	TCGG	.	PASS	VC=INDEL;AC=2;AF=0.25;AN=8;OLD_MULTIALLELIC=20:101:A/ACCA/TCGG	GT:AD:DP:GQ:PL	0/.:2,31:30:99:2407,697,574


Usage
=====

Assuming ``df_original`` is a variable of type DataFrame which contains the genomic variant records, an example of using this transformer for splitting multiallelic variants is:

.. tabs::

    .. tab:: Python

        .. code-block:: python

            df_split = glow.transform("split_multiallelics", df_original)

        .. invisible-code-block: python

            from pyspark.sql import Row

            expected_normalized_variant = Row(contigName='chr20', start=268, end=269, names=[], referenceAllele='A', alternateAlleles=['ATTTGAGATCTTCCCTCTTTTCTAATATAAACACATAAAGCTCTGTTTCCTTCTAGGTAACTGG'], qual=30.0, filters=[], splitFromMultiAllelic=False, INFO_AN=4, INFO_AF=[1.0], INFO_AC=[1], genotypes=[Row(sampleId='CHMI_CHMI3_WGS2', alleleDepths=None, phased=False, calls=[1, 1]), Row(sampleId='CHMI_CHMI3_WGS3', alleleDepths=None, phased=False, calls=[1, 1])])
            assert rows_equal(df_normalized.head(), expected_normalized_variant)

    .. tab:: Scala

        .. code-block:: scala

            df_split = Glow.transform("split_multiallelics", df_original)

.. notebook:: .. etl/splitmultiallelics-transformer.html
  :title: Splitting Multiallelic Variants notebook
