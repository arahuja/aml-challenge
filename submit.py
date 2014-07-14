import pandas as pd


def predict_submission_file(model,
                            test_file, transformer = None,
                            id_column = 'ID',
                            prediction_column = 'Class',
                            output_file = None,
                            compute_confidence = None):

    df = pd.read_csv(test_file)

    if transformer:
        X = transformer.transform(df)
    else:
        X = df

    predict_submission(model, df[id_column],
                       X, id_column,
                       prediction_column,
                       output_file)


def predict_submission(model,
                       id_idx,
                       X, id_column = 'ID',
                       prediction_column = 'Class',
                       output_file = None):
    p = model.predict_proba(X).T[-1]

    submission = pd.DataFrame({id_column: id_idx, prediction_column: p})

    if output_file:
        create_submission(submission, output_file)


def create_submission(submit_df,
                      output_file = 'submission.csv'):
    submit_df.sort_index(axis=1, inplace=True)
    submit_df.to_csv(output_file, index=False)
