import time
import pandas as pd
import numpy as np
from datetime import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv("../data/train/events.csv")
    mortality = pd.read_csv("../data/train/mortality_events.csv")

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    
   #events.groupby(['patient_id']).groups.keys(alive_patients)
       #count1 = 0
    #for patient in alive_patients:
     #   data = events[events.patient_id == patient]
     #   count1 = count1 + len(np.ravel(data["event_id"]))
    
    #count2 = 0
    #for patient in dead_patients:
     #   data = events[events.patient_id == patient]
     #   count2 = count2 + len(np.ravel(data["event_id"]))
     
    #data = pd.DataFrame()
    #for patient in alive_patients:
    #    data = data.append((events[events.patient_id == patient]))
    
    #num_events = np.ravel(data["event_id"])
    #num_events = num_events.tolist()
    #num_alive_max = num_events.count('LAB3009542')
    
    #num_alive_max = num_events.count(max(num_events,key=num_events.count))
    #num_alive_min = num_events.count(min(num_events,key=num_events.count))

    #num_alive_min = num_events.count('DIAG372448')


    #data = pd.DataFrame()
    #for patient in dead_patients:
    #    data = data.append((events[events.patient_id == patient]))
        
    #num_dead_events = np.ravel(data["event_id"])
    #num_dead_events = num_dead_events.tolist()
    
    #num_dead_max = num_dead_events.count(max(num_dead_events,key=num_dead_events.count))
    #num_dead_min = num_dead_events.count(min(num_dead_events,key=num_dead_events.count))
   
    
    patients = np.unique(events["patient_id"].values.ravel())
    dead_patients = np.unique(mortality["patient_id"].values.ravel())
    alive_patients = np.setdiff1d(patients, dead_patients)
    
    alive_events = events.loc[events["patient_id"].isin(alive_patients)]
    dead_events = events.loc[events["patient_id"].isin(dead_patients)]
    
    num_alive_events = alive_events["event_id"].count()
    num_dead_events = dead_events["event_id"].count()
    
    num_alive_max = alive_events["event_id"].value_counts().max()
    num_alive_min = alive_events["event_id"].value_counts().min()
    
    num_dead_max = dead_events["event_id"].value_counts().max()
    num_dead_min = dead_events["event_id"].value_counts().min()
        
    avg_dead_event_count = float(num_dead_events/len(dead_patients))
    max_dead_event_count = num_dead_max
    min_dead_event_count = num_dead_min
    avg_alive_event_count = float(num_alive_events/len(alive_patients))
    max_alive_event_count = num_alive_max
    min_alive_event_count = num_alive_min
    
    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
   '''
   
    patients = np.unique(events["patient_id"].values.ravel())
    dead_patients = np.ravel(mortality["patient_id"])
    alive_patients = np.setdiff1d(patients, dead_patients)
    
    alive_events = events.loc[events["patient_id"].isin(alive_patients)]
    dead_events = events.loc[events["patient_id"].isin(dead_patients)]
    
    #alive_events["timestamp"] = alive_events["timestamp"].apply(dateutil.parser.parse, dayfirst=True)
    
    
    sum_alive_enc = alive_events.groupby(["patient_id"])["timestamp"].nunique().sum()
    max_alive_enc = alive_events.groupby(["patient_id"])["timestamp"].nunique().max()
    min_alive_enc = alive_events.groupby(["patient_id"])["timestamp"].nunique().min()
    
    sum_dead_enc = dead_events.groupby(["patient_id"])["timestamp"].nunique().sum()
    max_dead_enc = dead_events.groupby(["patient_id"])["timestamp"].nunique().max()
    min_dead_enc = dead_events.groupby(["patient_id"])["timestamp"].nunique().min()
    
    
    #time_stamp1 =[]
    #for patient in alive_patients:
     #   data = events[events.patient_id == patient]
     #   time_stmp = np.unique(data["timestamp"].values.ravel())
     #   time_stmp = time_stmp.tolist()
     #   time_stamp1.append(time_stmp)
    #return time_stamp   
    #max_alive_enc = time_stamp1.count(max(time_stamp1, key=time_stamp1.count))
    #return max_alive_enc
            
    #time_stamp2 =[]
    #for patient in dead_patients:
     #   data = events[events.patient_id == patient]
     #   time_stmp = np.unique(data["timestamp"].values.ravel())
     #   time_stmp = time_stmp.tolist()
     #   time_stamp2.append(time_stmp)
    #max_dead_enc = time_stamp2.count(max(time_stamp2, key=time_stamp2.count))
    #return max_dead_enc
    
    avg_dead_encounter_count = sum_dead_enc/len(dead_patients)
    max_dead_encounter_count = max_dead_enc
    min_dead_encounter_count = min_dead_enc 
    avg_alive_encounter_count = sum_alive_enc/len(alive_patients)
    max_alive_encounter_count = max_alive_enc
    min_alive_encounter_count = min_alive_enc

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    
    patients = np.unique(events["patient_id"].values.ravel())
    dead_patients = np.ravel(mortality["patient_id"])
    alive_patients = np.setdiff1d(patients, dead_patients)
    
    alive_events = events.loc[events["patient_id"].isin(alive_patients)]
    dead_events = events.loc[events["patient_id"].isin(dead_patients)]
    
    first_alive_dates = alive_events.groupby(["patient_id"])["timestamp"].first()
    first_alive_dates = np.ravel(first_alive_dates)
    last_alive_dates = alive_events.groupby(["patient_id"])["timestamp"].last()
    last_alive_dates = np.ravel(last_alive_dates)
    
    first_dead_dates = dead_events.groupby(["patient_id"])["timestamp"].first()
    first_dead_dates = np.ravel(first_dead_dates)
    last_dead_dates = dead_events.groupby(["patient_id"])["timestamp"].last()
    last_dead_dates = np.ravel(last_dead_dates)
    

    def days_between(d1, d2):
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)
    
    alive_duration = []
    for a,b in zip(first_alive_dates,last_alive_dates):
        alive_duration.append(days_between(a,b))
    
    dead_duration = []
    for a,b in zip(first_dead_dates,last_dead_dates):
        dead_duration.append(days_between(a,b))
    
    
    
    avg_dead_rec_len = sum(dead_duration)/len(dead_patients)
    max_dead_rec_len = max(dead_duration)
    min_dead_rec_len = min(dead_duration)
    avg_alive_rec_len = sum(alive_duration)/len(alive_patients)
    max_alive_rec_len = max(alive_duration)
    min_alive_rec_len = min(alive_duration)

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
