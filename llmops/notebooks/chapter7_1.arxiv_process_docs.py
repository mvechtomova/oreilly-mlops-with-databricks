# Databricks notebook source

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from arxiv_curator.config import ProjectConfig

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema
volume= cfg.volume

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}"
)
spark.sql(
    f"CREATE VOLUME IF NOT EXISTS "
    f"{catalog}.{schema}.{volume}"
)

# COMMAND ----------

# search for papers in arxiv in the cs.AI category
# https://arxiv.org/category_taxonomy
# interested in the AI category within computer science:
# cs.AI - Artificial Intelligence

# The expected format for date is [YYYYMMDDTTTT+TO+YYYYMMDDTTTT]
# were the TTTT is provided in 24 hour time to the minute,
# in GMT.
import arxiv
import time

client = arxiv.Client()
metadata_table = f"{catalog}.{schema}.arxiv_papers"

if spark.catalog.tableExists(metadata_table):
    start = str(
        spark.sql(f"""
        SELECT max(processed)
        FROM {metadata_table}
        """).collect()[0][0]
    )
else:
    start = time.strftime(
        "%Y%m%d%H%M", (time.gmtime(time.time() - 24 * 3600 * 3))
    )

end = time.strftime("%Y%m%d%H%M", time.gmtime())

# COMMAND ----------
search = arxiv.Search(
    query=f"cat:cs.AI AND submittedDate:[{start} TO {end}]"
)
papers = client.results(search)

# COMMAND ----------

# create delta table with information about papers,
# including the location of the PDF file in volume storage

import os

records = []
pdf_dir = (f"/Volumes/{catalog}/{schema}/{volume}/{end}")
os.makedirs(pdf_dir, exist_ok=True)

for paper in papers:
    paper_id = paper.get_short_id()

    # download PDF
    try:
        paper.download_pdf(
            dirpath=pdf_dir, filename=f"{paper_id}.pdf"
        )
        # collect metadata (keep datetime intact)
        records.append(
            {
                "paper_id": paper_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "published": int(paper.published.strftime("%Y%m%d%H%M")),
                "processed": int(f"{end}"),
                "volume_path": f"{pdf_dir}/{paper_id}.pdf",
            }
        )
        break
    except Exception:
        logger.warning(
            f"Paper {paper_id} was not succesfully processed."
                )
        pass
    time.sleep(3) # to avoid rate limiting

# COMMAND ----------

if len(records) > 0:
    metadata_schema = T.StructType(
        [
        T.StructField("paper_id", T.StringType(), False),
        T.StructField("title", T.StringType(), True),
        T.StructField(
            "authors", T.ArrayType(T.StringType()), True
        ),
        T.StructField("summary", T.StringType(), True),
        T.StructField("pdf_url", T.StringType(), True),
        T.StructField("published", T.LongType(), True),
        T.StructField("processed", T.LongType(), True),
        T.StructField("volume_path", T.StringType(), True),
        ]
    )

    # create DataFrame
    metadata_df = spark.createDataFrame(
        records, schema=metadata_schema).withColumn(
        "ingest_ts", F.current_timestamp()
    )

    # write to UC
    metadata_df.write.format("delta").mode("append").saveAsTable(
        f"{metadata_table}")

    spark.sql(
        f"""
    CREATE TABLE IF NOT EXISTS
    {catalog}.{schema}.ai_parsed_docs (
    path STRING,
    parsed_content STRING,
    processed LONG)
    """
    )

    spark.sql(
        f"""
    INSERT INTO {catalog}.{schema}.ai_parsed_docs
    SELECT
    path,
    ai_parse_document(content) AS parsed_content,
    {end} AS processed
    FROM READ_FILES(
    "{pdf_dir}",
    format => 'binaryFile')
    """
    )

# COMMAND ----------
import re
import json
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.functions import (
    col,
    concat_ws,
    explode,
    udf,
)


# UDF to extract chunks from parsed_content JSON
def extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
    parsed_dict = json.loads(parsed_content_json)
    chunks = []

    for element in parsed_dict.get("document", {}).get("elements", []):
        if element.get("type") == "text":
            chunk_id = element.get("id", "")
            content = element.get("content", "")
            chunks.append((chunk_id, content))
    return chunks

chunk_schema = ArrayType(
    StructType(
        [
            StructField("chunk_id", StringType(), True),
            StructField("content", StringType(), True),
        ]
    )
)
extract_chunks_udf = udf(extract_chunks, chunk_schema)


df = spark.table(f"{catalog}.{schema}.ai_parsed_docs").where(
    f"processed = {end}"
)


def extract_paper_id(path):
    return path.replace(".pdf", "").split("/")[-1]

extract_paper_id_udf = udf(extract_paper_id, StringType())


# UDF to clean chunk text
def clean_chunk(text: str) -> str:
    # fix hyphenation across line breaks:
    # "docu-\nments" => "documents"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # collapse internal newlines into spaces
    t = re.sub(r"\s*\n\s*", " ", t)

    # collapse repeated whitespace
    t = re.sub(r"\s+", " ", t)

    return t.strip()


clean_chunk_udf = udf(clean_chunk, StringType())

# COMMAND ----------

# Load metadata table
metadata_df = spark.table(metadata_table).select(
    col("paper_id"),
    col("title"),
    col("summary"),
    concat_ws(", ", col("authors")).alias("authors"),
    (col("published") / 100000000).cast("int").alias("year"),
    ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
    ((col("published") % 1000000) / 10000).cast("int").alias("day"),
)

# Create the transformed table
chunks_df = (
    df.withColumn("paper_id", extract_paper_id_udf(col("path")))
    .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
    .withColumn("chunk", explode(col("chunks")))
    .select(
        col("paper_id"),
        col("chunk.chunk_id").alias("chunk_id"),
        clean_chunk_udf(col("chunk.content")).alias("text"),
        concat_ws(
            "_", col("paper_id"), col("chunk.chunk_id")
        ).alias("id"),
    )
    .join(metadata_df, "paper_id", "left")
)

# COMMAND ----------
# Write to table
chunks_table = f"{catalog}.{schema}.arxiv_chunks"
chunks_df.write.mode("append").saveAsTable(chunks_table)

# COMMAND ----------
spark.sql(
    f"""ALTER TABLE {chunks_table}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)
