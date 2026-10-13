from  pathlib import Path
import pandas as pd
from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers
from cleaning import build_normalizer

current_dir=Path(__file__).parent
cleaned_data_file=[
    current_dir/"output"/"phase1_train_filtered_cleaned_version",
    current_dir/"output"/"phase2_train_filtered_cleaned_version"
]

out_path=current_dir/"tokenizer.json"
vocab_size_inside_tokenizer=32000
min_frequency=2

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
    "[AR]", "[EN]",
    "[SYS]", "[USER]", "[ASST]"
]


def build_tokenizer():
    tokenizer=Tokenizer(
        models.BPE(unk_token="[UNK]")
    )
    Tokenizer.normalizer=build_normalizer()
    tokenizer.pre_tokenizer=pre_tokenizers.Metaspace(prepend_scheme="always")
    #prepend_scheme="always" gives an indication for the beging of the word _hello _world
    tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")

    return tokenizer

def corpus_iterator(df,batch_size=1000):
    texts=df["text"].tolist()
    for i in range(0,len(texts),batch_size):
        yield texts[i:i+batch_size]


def train(tokenizer,df):
    trainer=trainers.BpeTrainer(
        vocab_size=vocab_size_inside_tokenizer,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )
    tokenizer.train_from_iterator(
        corpus_iterator(df=df),
        trainer=trainer,
        length=len(df)
    )

    return tokenizer



def main():
    print("step 1 accessing data from files")

    frames=[]

    for file in cleaned_data_file:
        frame=pd.read_parquet(file)
        frames.append(frame)

    df=pd.concat(frames,ignore_index=True)

    tokenizer=build_tokenizer()
    tokenizer=train(tokenizer=tokenizer,df=df)
    print("SANITY CHECK: encode / decode bilingual examples")

    examples = [
        "الجمهورية الجزائرية عاصمتها الجزائر",
        "احمد ابراهيم ادم ولد في مصر",
        "The United States and the United Nations signed the agreement",
        "machine learning models require large amounts of training data",
    ]

    for text in examples:
        encoded_text = tokenizer.encode(text, add_special_tokens=False)
        decoded_text = tokenizer.decode(encoded_text.ids)

        print(f"\n input  : {text}")

        print(f" tokens : {encoded_text.tokens}")
        print(f" ids    : {encoded_text.ids}")
        print(f" decoded: {decoded_text}")



    return tokenizer.save(str(out_path))

if __name__=="__main__":
    main()
