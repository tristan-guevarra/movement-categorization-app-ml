# main method of the categorization app
# guevarra, tristan 
# 2nd year computer engineering

import pandas as pd
import numpy as np
import h5py


#Read each members data for storage in hdf5
cole_walk = pd.read_csv('Walking_Cole/Accelerometer.csv')
cole_jump = pd.read_csv('Jumping_Cole/Accelerometer.csv')
cole_walk['Label'] = 0
cole_jump['Label'] = 1

james_walk = pd.read_csv('Walking_James/Accelerometer.csv')
james_jump = pd.read_csv('Jumping_James/Accelerometer.csv')
james_walk['Label'] = 0
james_jump['Label'] = 1

tristan_walk = pd.read_csv('Walking_Tristan/Accelerometer.csv')
tristan_jump = pd.read_csv('Jumping_Tristan/Accelerometer.csv')
tristan_walk['Label'] = 0
tristan_jump['Label'] = 1


# Read walking data from CSV files
walking_data = pd.concat([cole_walk, james_walk, tristan_walk])

# Read jumping data from CSV files
jumping_data = pd.concat([cole_jump, james_jump, tristan_jump])


# Concatenate walking and jumping data
original_data = pd.concat([walking_data, jumping_data])


# Split the data into 5 second windows
window_data = np.array_split(original_data, range(0, len(original_data), 500))

# Shuffle the data
np.random.shuffle(window_data)

final = pd.concat(window_data)


# Split data into training (90%) and testing (10%) sets
train_size = int(0.9 * len(final))
training_set, testing_set = final[:train_size], final[train_size:]

testing_set.to_csv('testingset.csv')

with h5py.File('./hdf5_data.h5', 'w') as hdf:

    dataset_group = hdf.create_group('dataset')
    cole_group = hdf.create_group('cole')
    james_group = hdf.create_group('james')
    tristan_group = hdf.create_group('tristan')

    train_set = dataset_group.create_group('train')
    test_set = dataset_group.create_group('test')


    train_set.create_dataset('trainset', data=training_set)
    test_set.create_dataset('testset', data=testing_set)

    #Store the original data in the database
    cole_group.create_dataset('data', data = pd.concat([cole_walk, cole_jump]))
    james_group.create_dataset('data', data=pd.concat([james_walk, james_jump]))
    tristan_group.create_dataset('data', data=pd.concat([tristan_walk, tristan_jump]))
