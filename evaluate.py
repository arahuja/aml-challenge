import pandas as pd
from transformer import Transformer

import argparse
import logging
from model import predict_remission, predict_remission_length, predict_survival_time
from submit import predict_submission, create_submission
from scorers import get_top_features

if __name__ == '__main__':

    logging.basicConfig(format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', default=False, action='store_true', dest='submit')
    parser.add_argument('--preconcat', default=False, action='store_true', dest='preconcat')

    parser.add_argument('--eval', default=False, action='store_true', dest='eval')
    parser.add_argument('--linear', default=False, action='store_true', dest='linear')
    parser.add_argument('--bin', default=False, action='store_true', dest='bin')
    parser.add_argument('--challenge', default="all", dest='challenge')
    
    
    parser.add_argument('--scale', default=False, action='store_true', dest='scale')
    parser.add_argument('--print-coef', default=False, action='store_true', dest='print_coef')
    parser.add_argument('--output', default='c1_model.pkl', dest='output')
    args = parser.parse_args()

    train_data = pd.read_csv('trainingData-release.csv')
    submit_data = pd.read_csv('scoringData-release.csv')
    id_column = '#Patient_id'

    data = pd.concat([train_data, submit_data], ignore_index = True)
    transformer = Transformer(include_binned = args.bin, scale = args.scale)


    if args.challenge == 'all' or args.challenge == '1':
        logging.info("Running challenge 1 - Predict remission vs resistance")
        X_full = transformer.fit_transform(data)
        X = X_full[:len(train_data)]
        X_test = X_full[len(train_data):]

        # Challenge 1
        c1_target = train_data['resp.simple'].map(lambda x: 1 if x == 'CR' else 0)
        remission_model = predict_remission(X, c1_target, X_test, linear = args.linear)
        if args.print_coef:
            get_top_features(transformer.feature_names, remission_model)
        remission_prob = remission_model.predict_proba(X_test).T[-1]
        c1_df = pd.DataFrame({id_column: submit_data[id_column], 'CR_Confidence': remission_prob})
        create_submission(c1_df, "challenge1_submission.csv")
     
    if args.challenge == 'all' or args.challenge == '3':    
        # Challenge 3
        c3_target = train_data['Overall_Survival']
        logging.info("Running challenge 3 - Predict survival time for all patients")
        survival_time_model = predict_survival_time(X, c3_target, X_test)
        survival_time = survival_time_model.predict(X_test).T[-1]
        c3_df = pd.DataFrame({id_column: submit_data[id_column],
                            'Overall_Survival': survival_time,
                            'Confidence' : survival_time})
        create_submission(c3_df, 'challenge3_submission.csv')
        
    if args.challenge == 'all' or args.challenge == '2':
        # Challenge 2
        logging.info("Running challenge 2 - Predict remission length")
        c2train_data = data[data['resp.simple'] == 'CR']
        c2_data = pd.concat([c2train_data, submit_data], ignore_index=True)
        c2X_full = transformer.fit_transform(c2_data)
        c2X = X_full[:len(c2train_data)]
        c2X_test = X_full[len(c2train_data):]

        transformer = Transformer(include_binned = args.bin, scale = args.scale)

        c2_target = c2train_data['Remission_Duration']
        remission_length_model = predict_remission_length(remission_model, c2X, c2_target, c2X_test)
        remission_length = remission_length_model.predict(c2X_test).T[-1]
        print remission_length_model.predict(c2X_test)
        c2_df = pd.DataFrame({id_column: submit_data[id_column], 
                            'Remission_Duration': remission_length, 
                            'Confidence' : remission_length})
        create_submission(c2_df, 'challenge2_submission.csv')

