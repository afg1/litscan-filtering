import polars as pl
import guidance
from guidance import system, user, assistant, gen, select, models
import time
import click

system_prompt = ("You are an expert biocurator working in non-coding RNA. "
            "You are experienced with reading and evaluating papers based on their title and "
            "the content of their abstracts. You reply with short, to the point answers and do not "
            "use excessive words."
)

evaluation_prompt = ("Read the following title and abstract: \n"
                     "Title: {title}\n"
                     "Abstract: {abstract}\n"
                     "Using an automated tool, we found the following sentences mentioning ncRNA IDs in the paper:\n"
                     "{hits}\n"
                     "Do you think this paper is likely to be relevant to ncRNA, or irrelevant?\n"
                    #  "Remember, we are looking for papers about non-coding RNA, "
                    # "though papers about ncRNA interactions with proteins should be considered relevant\n"
                     )

@guidance
def evaluate_paper(lm, title, abstract, hits):
    formatted_prompt = evaluation_prompt.format(title=title, abstract=abstract, hits="\n".join(hits))
    with user():
        lm += formatted_prompt
    
    with assistant():
        lm += ("I believe the paper is likely to be "
               + select(name="judgement", options=["relevant", "irrelevant"])
            )
    print(lm)    

    return lm

def wrap_prefilter(row, lm):
    lm += evaluate_paper(row['title'], row['abstract'], row['sentence'])
    if lm['judgement'] == "relevant":
        return True
    return False


@click.command()
@click.argument("input_article_parquet")
@click.argument("input_sentence_parquet")
@click.argument("model_gguf")
@click.argument("output_prefix")
@click.option("--total_subset", default=1500)
@click.option("--train_test_val_split", default="60:20:20")
def main(input_article_parquet, input_sentence_parquet, model_gguf, output_prefix, total_subset, train_test_val_split):

    article_data = pl.scan_parquet(input_article_parquet).rename({"array_agg":"hits"})
    sentence_data =pl.scan_parquet(input_sentence_parquet)

    prefilter_subset = article_data.join(sentence_data, on="pmcid").collect(streaming=True).sample(total_subset)

    print(prefilter_subset)


    lm = models.LlamaCpp(model_gguf, n_gpu_layers=-1, n_ctx=8192, echo=False)

    start = time.time()
    prefilter_subset = prefilter_subset.with_columns(true_positive=pl.struct(pl.col("title"), pl.col("abstract"), pl.col("sentence")).map_elements(lambda x: wrap_prefilter(x, lm), return_dtype=bool))
    end = time.time()
    print(f"LLM judgement of {total_subset} title+abstract+hits took: {end-start} seconds")

    prefilter_subset.write_parquet(f"{output_prefix}llm_judged_all.parquet")

    split_fractions = [float(x)/100 for x in  train_test_val_split.split(":")]


    train = prefilter_subset.sample(fraction=split_fractions[0], with_replacement=False)
    test = prefilter_subset.sample(fraction=split_fractions[1], with_replacement=False)
    val = prefilter_subset.sample(fraction=split_fractions[2], with_replacement=False)
    
    train.write_parquet(f"{output_prefix}_train.parquet")
    test.write_parquet(f"{output_prefix}_test.parquet")
    val.write_parquet(f"{output_prefix}_val.parquet")


    print(train, test, val)

if __name__ == "__main__":
    main()