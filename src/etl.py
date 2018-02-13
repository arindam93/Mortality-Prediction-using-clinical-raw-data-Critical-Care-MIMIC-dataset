import utils
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    
    patients = np.unique(events["patient_id"].values.ravel())
    dead_patients = np.unique(mortality["patient_id"].values.ravel())
    alive_patients = np.setdiff1d(patients, dead_patients)
    
    alive_events = events.loc[events["patient_id"].isin(alive_patients)]
    
    indx_alive_dates = alive_events.groupby(["patient_id"], as_index=False)[["timestamp"]].max()
    
    indx_alive_dates["timestamp"] = indx_alive_dates.timestamp.apply(lambda x: utils.date_convert(x))
    #indx_alive_dates["timestamp"] = indx_alive_dates["timestamp"].apply(dateutil.parser.parse, dayfirst=False)
    indx_alive_dates["timestamp"] = indx_alive_dates["timestamp"].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    indx_dead_dates = pd.DataFrame()
    indx_dead_dates["patient_id"] = mortality["patient_id"]
    indx_dead_dates["timestamp"] = mortality.timestamp.apply(lambda x: 
        (datetime.strptime(x, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d"))
    
    
    indx_date = pd.concat([indx_alive_dates, indx_dead_dates]).reset_index().drop('index', 1)
    indx_date.columns = ["patient_id", "indx_date"]
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', index=False)
    
    indx_date['indx_date'] = indx_date.indx_date.apply(lambda x:
     datetime.strptime(x, "%Y-%m-%d"))
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    
    indx_date["lower_bound"] = indx_date.indx_date.apply(lambda x: 
        (x - timedelta(days=2000)))
        
    events["timestamp"] = events.timestamp.apply(lambda x: utils.date_convert(x))
    #events["timestamp"] = events["timestamp"].apply(dateutil.parser.parse, dayfirst=False)
    
    joint_events = pd.merge(events, indx_date, on='patient_id')
    filtered_events = joint_events[(joint_events["timestamp"] <= joint_events["indx_date"]) & (joint_events["timestamp"] >= joint_events["lower_bound"])]
    
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    
    event = pd.merge(filtered_events_df, feature_map_df,on = "event_id",how="left").dropna(subset=["value"])
#    event = event.groupby(['idx', 'patient_id', 'event_id']).agg({'value': [np.sum, np.mean, len, np.min, np.max]}).reset_index()
#    event['event_initials'] = event.event_id.apply(lambda x: x[:3])    
#    event['sum_values'] = event.loc[event['event_initials'] == 'LAB']['value']['len']
#    event.loc[event['event_initials'] == 'DRU', 'sum_values'] = list(
#    event.loc[event['event_initials'] == 'DRU']['value']['sum'])
#    event.loc[event['event_initials'] == 'DIA', 'sum_values'] = list(
#    event.loc[event['event_initials'] == 'DIA']['value']['sum'])
#    
#    feature_id = event['idx'].tolist()
#    value = event["sum_values"].tolist()
#    patient_id = event['patient_id'].tolist()
#    
#    all_list = [feature_id, value, patient_id]
#    
#    aggregated_events = pd.DataFrame()    
#    aggregated_events = pd.concat([aggregated_events, pd.DataFrame(all_list).T],axis=1).rename(columns={0:'feature_id',1:'value',2:'patient_id'}).reset_index()
        
    cols = list(event)
    cols[3], cols[1] = cols[1], cols[3]
    event = event.ix[:,cols]
    
    event = event.drop(["event_id"],axis=1)
        
    aggregated_events = event.groupby(["patient_id", "idx"])[["value"]].count().reset_index()
#   
    aggregated_events["max_feature_values"] = aggregated_events.groupby(
        "idx")["value"].transform(max)
    aggregated_events["feature_value"] = aggregated_events["value"].divide(
        aggregated_events["max_feature_values"], 0)
   
    aggregated_events = aggregated_events.rename(columns={"idx": "feature_id"})
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False) 

   
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)
    #print indx_date

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    #print filtered_events
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)
    #print aggregated_events

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    patient_features = {k: list(map(tuple, aggregated_events[["feature_id", "feature_value"]].values)) for k , aggregated_events in aggregated_events.groupby('patient_id')}
    
    #patient_features = {k: np.array(map(tuple, aggregated_events[["feature_id", "feature_value"]].values))) for k , aggregated_events in aggregated_events.groupby('patient_id')}

    #patient_features = {k: tuple((aggregated_events["feature_id"], aggregated_events["feature_value"])) for k, aggregated_events in aggregated_events.groupby(["patient_id"])}
    
    
    #patient_features = {k: tuple((aggregated_events["feature_id"], np.ravel(aggregated_events["feature_value"]).tolist())) for k, aggregated_events in aggregated_events.groupby(["patient_id"])}

    #patient_features = aggregated_events.groupby('patient_id')[['feature_id', 'feature_value']].apply(lambda x : [tuple(x) for x in x.values]).to_dict()
    
    
    #mortality = {"patient_id": aggregated_events["patient_id"], "value": ~(aggregated_events.patient_id.isin(dead_patients))*1}
    
    dead_patients = np.unique(mortality["patient_id"].values.ravel())
    
    pd.options.mode.chained_assignment = None
    events["value"] = 0
    events["value"][events.patient_id.isin(dead_patients)] = 1
    mortality_dic = events.set_index("patient_id")["value"].T.to_dict()
    
    
    
    
    #mortality = {k: ~((aggregated_events.patient_id.isin(dead_patients))*1) for k, aggregated_events in aggregated_events.groupby(["patient_id"])}
    

    return patient_features, mortality_dic

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    
    
#    
#    with deliverable1 as file:
#        
#        dict_content = ''
#        for k, v in patient_features.items():
#            dict_content = str(k) + ' ' + str(int(v[0][0])) + ':' + str("{:.4f}".format(v[0][1])) + ' '
#            deliverable1.write(dict_content)
#            deliverable1.write('\n')
            
            
    
    for key in patient_features:
        w_str = ''
        if mortality[key] == 1.0:
            w_str = w_str + '1 '
        else:
            w_str = w_str + '0 '
        
        for tuples in patient_features[key]:
            w_str = w_str + str(int(tuples[0])) + ':' + str("{:.4f}".format(tuples[1])) + ' '
        
        deliverable1.write(w_str)
        deliverable1.write('\n')

    deliverable2 = open(op_deliverable, 'w')
    
    for key in patient_features:
        w_str = str(int(key)) + ' '
        
        if mortality[key] == 1.0:
            w_str = w_str + '1.0 '
        else:
            w_str = w_str + '0.0 '
        
        for tuples in patient_features[key]:
            w_str = w_str + str(int(tuples[0])) + ':' + str("{:.4f}".format(tuples[1])) + ' '
        
        deliverable2.write(w_str)
        deliverable2.write('\n')
    
  
    deliverable1.close()
    deliverable2.close()

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()