import pandas as pd

def create_submission(model,
    test_file, id_column = 'ID',
    prediction_column = 'Class',
    transformer = None,
    output_file = 'submission.csv'):
  
  submission_test = pd.read_csv(test_file)
  print 

  if transformer:
    X = transformer.transform(submission_test)
  
  predictions = [x[1] for x in model.predict_proba(X)]

  submission = pd.DataFrame({id_column: submission_test[id_column], prediction_column: predictions})
  submission.sort_index(axis=1, inplace=True)
  submission.to_csv(output_file, index=False)