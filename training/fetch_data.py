import polars as pl
import os

connstr = os.getenv("PGDATABASE")

query = """
select lsa.pmcid, max(title) as title, max(abstract) as abstract, array_agg(job_id) from litscan_article lsa
join litscan_result lsr on lsr.pmcid = lsa.pmcid

group by lsa.pmcid
"""

sentence_query = """
select lsr.pmcid, array_agg(sentence) as sentence from litscan_result lsr
join litscan_abstract_sentence lsas on lsas.result_id = lsr.id
group by lsr.pmcid
union 
select lsr.pmcid, array_agg(sentence) as sentence from litscan_result lsr
join litscan_body_sentence lsbs on lsbs.result_id = lsr.id
where location != 'other'
group by lsr.pmcid"""

df = pl.read_database_uri(query, connstr)
df.write_parquet("litscan_all_hit_articles.parquet")

sentences = pl.read_database_uri(sentence_query, connstr)
sentences.write_parquet("litscan_all_hit_sentences.parquet")
