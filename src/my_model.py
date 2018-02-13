import utils
import pandas as pd 
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier


from sklearn.metrics import *
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
    
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    test_events = pd.read_csv("../data/test/events.csv")
    feature_map = pd.read_csv("../data/test/event_feature_map.csv")

    
    event = pd.merge(test_events, feature_map,on = "event_id",how="left").dropna(subset=["value"])
    cols = list(event)
    cols[5], cols[1] = cols[1], cols[5]
    event = event.ix[:,cols]
    
    event = event.drop(["event_id"],axis=1)
    
    aggregated_events = event.groupby(["patient_id", "idx"])[["value"]].count().reset_index()
   
    aggregated_events["max_value_by_feature"] = aggregated_events.groupby(
        "idx")["value"].transform(max)
    aggregated_events["normalized_value"] = aggregated_events["value"].divide(
        aggregated_events["max_value_by_feature"], 0)
    
    aggregated_events = aggregated_events.rename(columns={"normalized_value": "feature_value", "idx": "feature_id"})
    
    patient_features = {k: list(map(tuple, aggregated_events[["feature_id", "feature_value"]].values)) for k , aggregated_events in aggregated_events.groupby('patient_id')}


    deliverable1 = open('../data/test/feature_svmlight.test', 'wb')
                    
    for key in patient_features:
        w_str = '0 '
               
        for tuples in patient_features[key]:
            w_str = w_str + str(int(tuples[0])) + ':' + str("{:.4f}".format(tuples[1])) + ' '
        
        deliverable1.write(w_str)
        deliverable1.write('\n')

        
    deliverable2 = open('../deliverables/test_features.txt', 'w')

    for key in patient_features:
        w_str = str(int(key)) + ' '
        
        for tuples in patient_features[key]:
            w_str = w_str + str(int(tuples[0])) + ':' + str("{:.4f}".format(tuples[1])) + ' '
        
        deliverable2.write(w_str)
        deliverable2.write('\n')
        
    deliverable1.close()
    deliverable2.close()

    X_test, Y_test = utils.get_data_from_svmlight("../data/test/feature_svmlight.test")
    
    return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
    
    dec_tree_pred1 = BaggingClassifier(n_estimators=80, random_state=545510477)
    dec_tree_pred2 = AdaBoostClassifier(n_estimators=45, random_state=545510477)
    dec_tree_pred = VotingClassifier(estimators=[('bag', dec_tree_pred1), ('ada', dec_tree_pred2)], voting='soft', weights=[1,5])
    dec_tree_pred.fit(X_train.toarray(), Y_train)

    return dec_tree_pred.predict(X_test.toarray())


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	