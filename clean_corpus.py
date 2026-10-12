from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from cleaning import clean_dataframe,normalize_text,prepare_document
repo="bakrianoo/jabarti-llm-dataset"
max_chunks=20
HF_FILES = {
    "phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
    "phase1_eval":  "pretrain/phase1_eval-00000-of-00001.parquet",

    "phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
    "phase2_eval":  "pretrain/phase2_eval-00000-of-00001.parquet",

    "ft_train":     "finetune/train-00000-of-00001.parquet",
    "ft_eval":      "finetune/eval-00000-of-00001.parquet",
}

train_splits={"phase1_train","phase2_train"}
eval_splits={"phase1_eval","phase2_eval"}

out_dir=Path(__file__).parent/"output"

def download_and_filter():
    out_dir.mkdir(parents=True,exist_ok=True)
    for name, file in HF_FILES.items():
        local=hf_hub_download(
            repo_id=repo,
            filename=file,
            repo_type="dataset"

        )
        df=pd.read_parquet(local)

        if name in train_splits:
            before_filter=len(df)
            df=(
                df.sort_values(["article_id","chunk_index"])
                .groupby("article_id")
                .head(max_chunks)
                .reset_index(drop=True)
            )
            after_filter=len(df)
            print(f"before filteration {before_filter} aftet filteration {after_filter}")

        out_path=out_dir / f"{name}_filtered_version"
        df.to_parquet(out_path,index=False)



def clean_phase(phase,input_path,output_path):
    df=pd.read_parquet(path=input_path)

    print(f"input_path : {input_path}")
    print(f"N rows      :{df.shape[0]}")

    cleaned_df=clean_dataframe(df)

    removed_rows=df.shape[0]-cleaned_df.shape[0]
    print(f"Rows kept :{len(cleaned_df)} dropped rows:{removed_rows}")

    cleaned_df.to_parquet(output_path,index=False)





def main():
    download_and_filter()
    for name in train_splits:
        input_path=out_dir / f"{name}_filtered_version"
        output_path=out_dir / f"{name}_filtered_cleaned_version"
        clean_phase(phase=name,input_path=input_path,output_path=output_path)

    for name in eval_splits:
            input_path=out_dir / f"{name}_filtered_version"
            output_path=out_dir / f"{name}_filtered_cleaned_version"
            clean_phase(phase=name,input_path=input_path,output_path=output_path)
    



if __name__=="__main__":
    main()