from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

spark.sql("""
    SELECT DISTINCT
        billing_origin_product,
        regexp_replace(
            sku_name, '_(US_EAST|EU_WEST|US_EAST_2|US_WEST|US_WEST_2)$', ''
        ) AS sku_name,
        usage_unit
    FROM system.billing.usage
    WHERE billing_origin_product IN (
        'VECTOR_SEARCH', 'AI_FUNCTIONS', 'MODEL_SERVING',
        'FOUNDATION_MODEL_APIS', 'AI_GATEWAY', 'LAKEBASE'
    )
    ORDER BY billing_origin_product, sku_name
""").display()
