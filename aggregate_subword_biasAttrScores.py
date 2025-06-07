import argparse
import pandas as pd
import numpy as np

def aggregate_subwords(df, subword_prefix):
    matching_words_list = []
    words_sjsd_list = []
    sorted_words_sjsd_list = []
    sorted_word_contributions_list = []

    for i in range(len(df)):
        tokens = df.loc[i, 'matching_tokens']
        scores = df.loc[i, 'tokens_sjsd']

        words = []
        words_sjsd = []

        current_word = ""
        current_scores = []

        for j in range(len(tokens)):
            token = tokens[j]
            score = scores[j]

            if j == 0 or token.startswith(subword_prefix):
                if current_word:
                    words.append(current_word)
                    words_sjsd.append(np.mean(current_scores))
                current_word = token.replace(subword_prefix, '')
                current_scores = [score]
            else:
                current_word += token
                current_scores.append(score)

        if current_word:
            words.append(current_word)
            words_sjsd.append(np.mean(current_scores))

        sorted_indices = np.argsort(words_sjsd)
        sorted_words_sjsd = [words_sjsd[i] for i in sorted_indices]
        sorted_word_contributions = [words[i] for i in sorted_indices]

        matching_words_list.append(words)
        words_sjsd_list.append(words_sjsd)
        sorted_words_sjsd_list.append(sorted_words_sjsd)
        sorted_word_contributions_list.append(sorted_word_contributions)

    df['matching_words'] = matching_words_list
    df['words_sjsd'] = words_sjsd_list
    df['sorted_words_sjsd'] = sorted_words_sjsd_list
    df['sorted_word_contributions'] = sorted_word_contributions_list

    return df

def main():
    parser = argparse.ArgumentParser(description='Aggregate subword bias attribution scores in a CSV.')
    parser.add_argument('csv_file', help='Path to the CSV file to process')
    parser.add_argument('--prefix', default='Ġ', help='Subword prefix character (default: Ġ). Use ▁ for SEALION-3B; Ġ for SEALLM7B-Chat, RoBERTa-Tagalog, and GPT-2.')
    parser.add_argument('--output', default='sorted_output.csv', help='Output CSV file name (default: sorted_output.csv)')

    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    df['matching_tokens'] = df['matching_tokens'].apply(eval)
    df['tokens_sjsd'] = df['tokens_sjsd'].apply(eval)

    df = aggregate_subwords(df, args.prefix)
    df.to_csv(args.output, index=False)
    print(f"Processed CSV saved to {args.output}")

if __name__ == '__main__':
    main()
