import json
import os
import re
import time

import arxiv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from arxiv_curator.config import ProjectConfig


class DataProcessor:
    """
    DataProcessor handles the complete workflow of:
    - Downloading papers from arXiv
    - Storing paper metadata
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume

        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())
        self.pdf_dir = f"/Volumes/{self.catalog}/{self.schema}\
            /{self.volume}/{self.end}"
        os.makedirs(self.pdf_dir, exist_ok=True)
        self.papers_table = f"{self.catalog}.{self.schema}.arxiv_papers"
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"

    def _get_range_start(self) -> tuple[str, str]:
        """
        Get start time range for arxiv paper search.
        If arxiv_papers table exists, uses max(processed) as start.
        Otherwise, uses 3 days ago as start.

        Returns:
            start string in "YYYYMMDDHHMM" format
        """

        if self.spark.catalog.tableExists(self.papers_table):
            result = self.spark.sql(f"""
                SELECT max(processed)
                FROM {self.apapers_table}
            """).collect()
            start = str(result[0][0])
            logger.info(
                f"Found existing arxiv_papers table. Starting from: {start}"
            )
        else:
            start = time.strftime(
                "%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * 3)
            )
            logger.info(
                f"""No existing arxiv_papers table.
                Starting from 3 days ago: {start}"""
            )
        return start

    def download_and_store_papers(
        self,
    ) -> list[dict] | None:
        """
        Download papers from arxiv and store metadata
        in arxiv_papers table.

        Returns:
            List of paper metadata dictionaries if papers were downloaded,
            otherwise None
        """
        start = self._get_range_start()

        # Search for papers in arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"cat:cs.AI AND submittedDate:[{start} TO {self.end}]"
        )
        papers = client.results(search)

        # Download papers and collect metadata
        records = []

        for paper in papers:
            paper_id = paper.get_short_id()
            try:
                paper.download_pdf(
                    dirpath=self.pdf_dir, filename=f"{paper_id}.pdf"
                )
                # Collect metadata
                records.append(
                    {
                        "paper_id": paper_id,
                        "title": paper.title,
                        "authors": [
                            author.name for author in paper.authors
                        ],
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "published": int(
                            paper.published.strftime("%Y%m%d%H%M")
                        ),
                        "processed": int(self.end),
                        "volume_path": f"{self.pdf_dir}/{paper_id}.pdf",
                    }
                )
                break
            except Exception:
                logger.warning(
                        f"Paper {paper_id} was not successfully processed."
                    )
            # Avoid hitting API rate limits
            time.sleep(3)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new papers found.")
            return None

        logger.info(f"Downloaded {len(records)} papers to {self.pdf_dir}")

        # Create DataFrame and save to arxiv_papers table
        schema = T.StructType(
            [
                T.StructField("paper_id", T.StringType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField("summary", T.StringType(), True),
                T.StructField("pdf_url", T.StringType(), True),
                T.StructField("published", T.LongType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        metadata_df = self.spark.createDataFrame(
            records, schema=schema).withColumn(
            "ingest_ts", current_timestamp()
        )

        metadata_df.write.format("delta").mode("append").saveAsTable(
            self.papers_table)
        logger.info(
            f"Saved {len(records)} paper records to {self.papers_table}"
        )
        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
        """)

        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                path,
                ai_parse_document(content) AS parsed_content,
                {self.end} AS processed
            FROM READ_FILES(
                "{self.pdf_dir}",
                format => 'binaryFile'
            )
        """)

        logger.info(
            f"Parsed PDFs from {self.pdf_dir} and saved to {self.parsed_table}"
        )

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """
        Extract chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing 
            parsed document structure

        Returns:
            List of tuples containing (chunk_id, content)
        """
        parsed_dict = json.loads(parsed_content_json)
        chunks = []

        for element in parsed_dict.get("document", {}).get("elements", []):
            if element.get("type") == "text":
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                chunks.append((chunk_id, content))

        return chunks

    @staticmethod
    def _extract_paper_id(path: str) -> str:
        """
        Extract paper ID from file path.

        Args:
            path: File path (e.g., "/path/to/paper_id.pdf")

        Returns:
            Paper ID extracted from the path
        """
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text
        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean chunks.
        Reads from ai_parsed_docs table and saves to arxiv_chunks table.
        """
        logger.info(
            f"""Processing parsed documents from
            {self.parsed_table} for end date {self.end}"""
        )

        df = self.spark.table(self.parsed_table).where(
            f"processed = {self.end}"
        )

        # Define schema for the extracted chunks
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        extract_paper_id_udf = udf(self._extract_paper_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        metadata_df = self.spark.table(self.papers_table).select(
            col("paper_id"),
            col("title"),
            col("summary"),
            concat_ws(", ", col("authors")).alias("authors"),
            (col("published") / 100000000).cast("int").alias("year"),
            ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
            ((col("published") % 1000000) / 10000).cast("int").alias("day"),
)

        # Create the transformed dataframe
        chunks_df = (
            df.withColumn("paper_id", extract_paper_id_udf(col("path")))
            .withColumn(
                "chunks", extract_chunks_udf(col("parsed_content"))
            )
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("paper_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias(
                    "id"
                ),
            )
            .join(metadata_df, "paper_id", "left")
        )

        # Write to table
        arxiv_chunks_table = f"{self.catalog}.{self.schema}.arxiv_chunks_table"
        chunks_df.write.mode("append").saveAsTable(arxiv_chunks_table)
        logger.info(f"Saved chunks to {arxiv_chunks_table}")

        # Enable Change Data Feed
        self.spark.sql(f"""
            ALTER TABLE {arxiv_chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {arxiv_chunks_table}")

    def process_and_save(self) -> None:
        """
        Complete workflow: download papers, parse PDFs, and process chunks.
        Matches the logic from chapter6_1 notebook.
        """
        # Step 1: Download papers and store metadata
        records = self.download_and_store_papers()

        # Only continue if we have new papers
        if records is None:
            logger.info("No new papers to process. Exiting.")
            return

        # Step 2: Parse PDFs with ai_parse_document
        self.parse_pdfs_with_ai()
        logger.info("Parseed documents.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
