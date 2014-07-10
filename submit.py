import pandas as pd

def create_submission(model,
    test_file, id_column = 'ID',
    prediction_column = 'Class',
    transformer = None,
    output_file = 'submission.csv',
    compute_confidence = None):
  
  submission_test = pd.read_csv(test_file) 

  if transformer:
    X = transformer.transform(submission_test)
  
  predictions = model.predict_proba(X).T[0]

  submission = pd.DataFrame({id_column: submission_test[id_column], prediction_column: predictions})
  if confidence:
    submission['Confidence'] = 1

  submission.sort_index(axis=1, inplace=True)
  submission.to_csv(output_file, index=False)