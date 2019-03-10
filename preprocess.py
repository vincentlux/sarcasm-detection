# create train_label.txt train_comment.txt test_sent.txt
from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.tsv', sep='\t')
    submission_df = pd.read_csv('./data/test.tsv', sep='\t')
    # train_df['label'].to_csv('./data/train_label.txt', header=False, index=False, encoding='utf-8')
    # train_df['comment'].to_csv('./data/train_comment.txt', header=False, index=False, encoding='utf-8')
    # submission_df['comment'].to_csv('./data/submission_comment.txt', header=False, index=False, encoding='utf-8')

    # resplit
    train, val = train_test_split(train_df, test_size=0.1, random_state=42)
    train.to_csv('./data/train_resplit.tsv', sep='\t', index=False, encoding='utf-8')
    val.to_csv('./data/val_resplit.tsv', sep='\t', index=False, encoding='utf-8')

