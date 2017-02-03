# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:31:45 2017

@author: Chen Bowen
"""
import urllib
from numpy import * 
import numpy as np
import pandas as pd
from pylab import *
import csv
import glob
from scipy.ndimage import filters
from scipy.misc import imread
from scipy.misc import imresize,imsave
import matplotlib.pyplot as plt
import os

# Part 1

act = list(set([a.split("\t")[0] for a in open("faces_subset.txt").readlines()]))
act1 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def download_uncropped_images(input_file):
    testfile = urllib.URLopener()            
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    crop_coordinates = {}
    existed_url = {}
    actor_name = {}
    
    if not os.path.exists(os.getcwd() + '/uncropped'):
        os.makedirs(os.getcwd() + '/uncropped')
    else:
        pass
    
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(input_file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                if not line.split()[4] in existed_url.values():
                    timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                    crop_coordinates[filename] = line.split()[5]
                    existed_url[filename] = line.split()[4]
                    actor_name[filename] = line.split()[0] + ' ' + line.split()[1]
                    if not os.path.isfile("uncropped/"+filename):

                        continue
                else: 
                    continue
                
                print filename
                i += 1 
                                
    if not os.path.exists(os.getcwd() + '\\Namelist'):
         os.makedirs(os.getcwd() + '\\Namelist')
    else: 
        pass
               
    crop_coordinates_dataframe = pd.DataFrame.from_dict(crop_coordinates,orient='index')
    crop_coordinates_dataframe.columns = ['crop_coordinates']
    crop_coordinates_dataframe['image_name'] = crop_coordinates_dataframe.index
    crop_coordinates_dataframe.to_csv(os.getcwd() +'\\Namelist\crop coordinates.csv', index = False) 
                         
    actor_name_dataframe =  pd.DataFrame.from_dict(actor_name,orient='index')
    actor_name_dataframe.columns = ['actor_names']
    actor_name_dataframe['image_name'] = actor_name_dataframe.index
    actor_name_dataframe.to_csv(os.getcwd() + '\\Namelist\actor names.csv', index = False)
    return crop_coordinates

# Part 2

def get_all_images(paths):
    image_list = glob.glob(paths)
    return image_list

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def crop_and_gray_image(image_path,crop_coordinates):  
    image = imread(image_path)
    image_name = image_path.split('\\')[-1]
    
    x1 = int(crop_coordinates[image_name].split(',')[0])
    y1 = int(crop_coordinates[image_name].split(',')[1])
    x2 = int(crop_coordinates[image_name].split(',')[2])
    y2 = int(crop_coordinates[image_name].split(',')[3])
    cropped  = image[y1:y2, x1:x2]
    std_cropped = imresize(cropped, (32,32))
    std_gray_cropped = rgb2gray(std_cropped)
    imshow(std_gray_cropped)
        
    return [image_name, std_gray_cropped]

def update_images(image_list, crop_coordinates):
    if not os.path.exists(os.getcwd() + '/cropped'):
        os.makedirs(os.getcwd() + '/cropped')
    else:
        pass
    crop_path = os.getcwd() + '/cropped/'

    for i in image_list:
        try:
              plt.show()
              std_gray_cropped = crop_and_gray_image(i,crop_coordinates)[1]
              image_name = crop_and_gray_image(i,crop_coordinates)[0]
              imsave(crop_path + image_name, std_gray_cropped)
        except:
            pass     
    return 0

def clear_non_readable_files(actor_name_dataframe,image_list):
    image_name_list = []
    for i in image_list:
        image_name = i.split('\\')[-1]
        image_name_list.append(image_name)
    actor_dataframe = actor_name_dataframe[actor_name_dataframe['image_name'].isin(image_name_list)]    
    return [image_name_list, actor_dataframe]
    

def separate_training_sets(actor_name,actor_name_dataframe, image_name_list, cropped_path):
    
    image_paths = {}
    for i in image_name_list:
        image_paths[i] = cropped_path[:-1] + i 
    #image_paths_col = pd.Series(image_paths)
    image_paths_col = actor_name_dataframe['image_name'].map(image_paths).to_frame()
    image_paths_col.columns  = ["image_path"]
    actor_name_dataframe = actor_name_dataframe.join(image_paths_col)
    actor_image_df = actor_name_dataframe[actor_name_dataframe['actor_names'] == actor_name]
    
    try:
        actor_image_df_used = actor_image_df.sample(n = 120, random_state=1)
        training_set = actor_image_df_used.sample(n = 100, random_state=1)
        test_validation = actor_image_df_used.drop(training_set.index)
        test_set = test_validation.sample(n = 10, random_state=1)
        validation_set =  test_validation.drop(test_set.index)
    except:
        actor_image_df_used = actor_image_df
        training_set = actor_image_df_used.sample(n = 100, random_state=1)
        test_validation = actor_image_df_used.drop(training_set.index)
        test_set = test_validation.sample(n = 8, random_state=1)
        validation_set =  test_validation.drop(test_set.index)

    return [training_set, test_set, validation_set]
 
    
def get_subsets(actor_name, actor_name_dataframe, image_name_list, cropped_path):
    
    image_paths = {}
    for i in image_name_list:
        image_paths[i] = cropped_path[:-1] + i 
    image_paths_col = actor_name_dataframe['image_name'].map(image_paths).to_frame()
    image_paths_col.columns  = ["image_path"]
    actor_name_dataframe = actor_name_dataframe.join(image_paths_col)
    actor_image_df = actor_name_dataframe[actor_name_dataframe['actor_names'] == actor_name]
    
    return actor_image_df

def organize_subset_data(actor_list, actor_name_dataframe, image_name_list, cropped_path):
    organized_sub_data = {}
    for i in actor_list:
        organized_sub_data[i] = get_subsets(i,actor_name_dataframe, image_name_list, cropped_path)
        
    return organized_sub_data

def organize_all_data(actor_list, actor_name_dataframe, image_name_list, cropped_path):
    organized_data = {}
    for i in actor_list:
        organized_data[i] = separate_training_sets(i,actor_name_dataframe, image_name_list, cropped_path)
        
    return organized_data

# Part 3

def add_Bill_label(Bill_data):
    for i in Bill_data:
        i['label'] = 0  
    return Bill_data

def add_Steve_label(Steve_data):  
    for i in Steve_data:
        i['label'] = 1         
    return Steve_data

def build_regression_sets_two_actors(Bill_Data,Steve_Data):
    training_set =pd.concat([Bill_Data[0], Steve_Data[0]], axis=0)
    test_set = pd.concat([Bill_Data[1], Steve_Data[1]], axis=0)
    validation_set = pd.concat([Bill_Data[2], Steve_Data[2]], axis=0)
    regression_sets = [training_set, test_set, validation_set]
    return regression_sets

def reshape_image(image_path):
    image = imread(image_path)/255.0
    reshpaed_image = np.reshape(image,(1,1024))
    return reshpaed_image

def get_regression_parameters(sets):
    x = np.ones((1,1024))
    for j in sets['image_path']:
        reshaped_image = reshape_image(j)
        x = np.vstack((x, reshaped_image))
    A = [1] * np.shape(x[1:,:])[0]
    x= np.vstack((A, (x[1:,:]).T)) 
    y = sets['label']
    set_inputs = [x, y]
  
    return set_inputs   

        
def f(x, y, theta):
    return sum((y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    return -2*sum((y-dot(theta.T, x))*x, 1)

def gradient_descent(f, df, x, y, initial_theata, alpha, max_iter):  
    EPS = 1e-5   #EPS = 10**(-5)
    previous_theta = initial_theata-10*EPS
    theta = initial_theata.copy()
    theta_x_dim = int(np.shape(theta)[0])
    theta_y_dim = int(np.shape(theta)[1])
    iter  = 0
    while norm(theta - previous_theta) >  EPS and iter < max_iter:
        previous_theta = theta.copy()
        # match dimension of theta
        theta -= alpha*df(x, y, theta).reshape((theta_x_dim,theta_y_dim))
        if iter % 5000 == 0:
            print "Iter", iter
            print "f(x) = %.2f" % (f(x, y, theta)) 
            print "Gradient: ", df(x, y, theta), "\n"
        iter += 1
    print "End Iteration", iter
    return [theta, f(x, y, theta)]

def train_regression_model(training_sets, f, df, alpha):
    [train_x,train_y] = get_regression_parameters(training_sets)
    train_y = train_y.reshape(1,len(train_y))
    initial_theta  = np.array([0.0]*1025).reshape(1025,1)
    max_iter = 30000
    theta, cost_function_value = gradient_descent(f, df, train_x, train_y, initial_theta, alpha, max_iter)
    
    return [theta, cost_function_value]

def evaluate_performace(test_set,theta):
    [test_x,test_y] = get_regression_parameters(test_set)
    hypothesis_y_test = dot(theta.T,test_x)
    hypothesis_y_test[hypothesis_y_test>= 0.5] = 1
    hypothesis_y_test[hypothesis_y_test< 0.5] = 0
     
    test_y = test_y.reshape(1,len(test_y))
    cost_f = f(test_x,test_y, theta)
    total_correct = 0
    
    for i in range(np.shape(test_y)[1]):
        if hypothesis_y_test[0][i] == test_y[0][i]:
            correct = 1
        else:
            correct = 0
        total_correct += correct
        
    return [total_correct*1.0/np.shape(test_y)[1], cost_f]

# Part 4

def visualize_theta(theta, save_name):
    if not os.path.exists(os.getcwd() + '/results'):
         os.makedirs(os.getcwd() + '/results')
    else: 
        pass 
    theta_im = theta[1:].reshape(32,32)
    imshow(theta_im, cm.coolwarm)
    figure = plt.gcf()
    plt.title('Visualized Theta - Bill/Steve Classification')
    figure.savefig(os.getcwd() + '/results/' + save_name)
#    imsave(os.getcwd() + "\\results\\" + save_name, theta_im)
    return None    

def two_per_actor_training_set(Bill_Data,Steve_Data):
    training_set_two_actor =pd.concat([Bill_Data[0].sample(n = 2, random_state= 5), Steve_Data[0].sample(n = 2, random_state= 2353)], axis=0)
    return training_set_two_actor
    

# Part 5

def add_gender_labels(subset_data):
    male_actors = ['Gerard Butler', 'Michael Vartan', 'Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Daniel Radcliffe']
    female_actors = ['Angie Harmon', 'Fran Drescher', 'Lorraine Bracco', 'Peri Gilpin', 'America Ferrera', 'Kristin Chenoweth']
    for key in subset_data:
        if key in male_actors:
           subset_data[key]['label'] = 1
               
        elif key in female_actors:
            subset_data[key]['label'] = 0
                 
    return subset_data

def build_regression_sets_gender(gender_labeled_sets, sample_size_train):
    training_act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    training_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    validation_test_data = pd.DataFrame()
    for key in gender_labeled_sets:
        if key in training_act:
            training_data = pd.concat([training_data, gender_labeled_sets[key]])
            validation_data =  pd.concat([validation_data, gender_labeled_sets[key]])
        else:
            validation_test_data = pd.concat([validation_test_data, gender_labeled_sets[key]])
    try:
        training_male_data = training_data[training_data['label'] == 1].sample(n = sample_size_train/2, random_state=1)
        training_female_data = training_data[training_data['label'] == 0].sample(n = sample_size_train/2, random_state=1)
        training_data =  pd.concat([training_male_data, training_female_data])
        
        validation_data = validation_data.drop(training_data.index)
        validation_male_data =  validation_data[validation_data['label'] == 1].sample(n = 50, random_state=1)
        validation_female_data =  validation_data[validation_data['label'] == 0].sample(n = 50, random_state=1)
        validation_data =  pd.concat([validation_male_data, validation_female_data])
        
    
    except: 
        training_male_data = training_data[training_data['label'] == 1]
        training_female_data = training_data[training_data['label'] == 0]
        training_data =  pd.concat([training_male_data, training_female_data])
        validation_data = validation_data.drop(training_data.index)
        validation_male_data =  validation_data[validation_data['label'] == 1]
        validation_female_data =  validation_data[validation_data['label'] == 0]
        validation_data =  pd.concat([validation_male_data, validation_female_data])
    
    return [training_data, validation_data, validation_test_data]

def get_model_stats(train_regression_model, evaluate_performace, build_regression_sets_gender, f, df):
    sample_size_train = range(2, 810 , 20)
    model_stats = {}
    alpha = 0.000001
    for i in sample_size_train:   
        print "Testing n = ", i 
        training_data = build_regression_sets_gender(gender_labeled_sets, i)[0]
        validation_data = build_regression_sets_gender(gender_labeled_sets, i)[1]
        theta = train_regression_model(training_data, f, df, alpha)
        train_performance = evaluate_performace(training_data, theta)[0]
        validation_performance = evaluate_performace(validation_data, theta)[0]
        evaluate_results =  [train_performance, validation_performance]
        model_stats[i] = evaluate_results
    return model_stats 

def plot_model_stats(model_stats):
    training_performances = {}
    validation_performances = {}
    for key in model_stats:
        training_performances[key] = model_stats[key][0]
        validation_performances[key] = model_stats[key][1]
    training_list = sorted(training_performances.items())
    validation_list = sorted(validation_performances.items())
    n, accuracy_training = zip(*training_list)
    n, accuracy_validation = zip(*validation_list)
    plt.figure(figsize=(10,6))
    plt.axis([0, 1000, 0, 1.5])
    plt.title('Accuracy of Training Set and Test Set')
    plt.xlabel('Number of Images in Training Set n')
    plt.ylabel('Accuracy %')
    
    plt.plot(n, accuracy_training, label = 'Training Set Performances')
    plt.plot(n, accuracy_validation, label = 'Validation Set Performances')
    plt.legend(loc = 'upper left')
    figure = plt.gcf()
    plt.show()
    figure.savefig(os.getcwd() + '/results/Part 5 - Overfitting.png')
    return None
    
# Part 6

def f_multi_class(x, y, theta):
    return  sum((y - dot(theta.T,x)) ** 2) + sum(dot(theta.T,theta))

def df_multi_class(x, y, theta):
    return 2 * dot(x,(dot(theta.T, x) - y).T) + theta

def finite_difference(x, y, theta):
    delta_h = 0.0000001
    df_FD = np.zeros(np.shape(theta))
    for i in range(np.shape(theta)[0]):
        for j in range(np.shape(theta)[1]):          
            h = np.zeros(np.shape(theta))
            h[i][j] = delta_h
            df_FD[i][j] = (f_multi_class(x, y, theta+h) - f_multi_class(x,y, theta-h))/(2*delta_h)
    return df_FD

def one_hot_encode_set(subset_data):
    full_data = pd.DataFrame()
    act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    for key in subset_data:
        if key in act:
            full_data = pd.concat([full_data,subset_data[key]])
    
    encoded_cols =  pd.get_dummies(full_data['actor_names']) 
    one_hot_full_data = full_data.join(encoded_cols)    
    try:
       one_hot_full_data =  one_hot_full_data.drop('label', axis = 1)
       
    except:
        pass
    return one_hot_full_data

def separate_training_set_multiclass(actor_name, one_hot_full_data):
    actor_image_one_hot = one_hot_full_data[one_hot_full_data['actor_names'] == actor_name]
    try:
        actor_image_one_hot_used = actor_image_one_hot.sample(n = 120, random_state=1)
        training_set = actor_image_one_hot_used.sample(n = 100, random_state=1)
        test_validation = actor_image_one_hot_used.drop(training_set.index)
        test_set = test_validation.sample(n = 10, random_state=1)
        validation_set =  test_validation.drop(test_set.index)
    except:
        actor_image_one_hot_used = actor_image_one_hot
        training_set = actor_image_one_hot_used.sample(n = 100, random_state=1)
        test_validation = actor_image_one_hot_used.drop(training_set.index)
        test_set = test_validation.sample(n = 8, random_state=1)
        validation_set =  test_validation.drop(test_set.index)

    return [training_set, test_set, validation_set]

def build_regression_sets_multi_class(act_sub, separate_training_set_multiclass):
    organized_data_multi = {}
    for i in act_sub:
        organized_data_multi[i] = separate_training_set_multiclass(i, one_hot_full_data)
    training_df = pd.DataFrame()
    test_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    for key in organized_data_multi:
      training_df = pd.concat([training_df,organized_data_multi[key][0]])  
      test_df = pd.concat([test_df,organized_data_multi[key][1]])
      validation_df = pd.concat([validation_df,organized_data_multi[key][2]])
      
    return [training_df, test_df, validation_df]
    
def get_regression_parameters_multiclass(sets):     
    x = np.ones((1,1024))
    for j in sets['image_path']:
        reshaped_image = reshape_image(j)
        x = np.vstack((x, reshaped_image))
    A = [1] * np.shape(x[1:,:])[0]
    x= np.vstack((A, (x[1:,:]).T)) 
    ycols = sets.columns.values.tolist()[-6:]
    y = sets[ycols].as_matrix().T
    set_inputs = [x, y]
      
    return set_inputs   

# Part 7

def train_regression_model_multi(training_df, f_multi_class, df_multi_class, alpha):
    [training_x_multi, training_y_multi] = get_regression_parameters_multiclass(training_df)  
    initial_theta  = np.full((int(np.shape(training_x_multi)[0]) ,int(np.shape(training_y_multi)[0])),0.0)
    max_iter = 100000
    theta = gradient_descent(f_multi_class, df_multi_class, training_x_multi, training_y_multi, initial_theta, alpha, max_iter)[0]
    
    return theta

def evaluate_regression_model_multi(test_df,theta):
    [test_x,test_y] = get_regression_parameters_multiclass(test_df)
    predict_y_test = dot(theta.T,test_x).T
    predict_y_test[predict_y_test< 0.5] = 0                        
    hypothesis_y_test = np.zeros_like(predict_y_test)
    hypothesis_y_test[np.arange(len(predict_y_test)), predict_y_test.argmax(1)] = 1 
    total_correct = 0 
    for i in range(len(hypothesis_y_test)):
        if np.array_equal(test_y.T[i], hypothesis_y_test[i]):
            total_correct += 1
    return total_correct*1.0/np.shape(test_y)[1]
    
# Part 8    

def visualize_theta_multi(theta_multi, training_df):
    theta_im = theta_multi[1:]
    image_names = training_df.columns.values.tolist()[-6:]
    for i in range(np.shape(theta_im)[1]):

        theta_im1 = theta_im.T[i].reshape(1024,1)
        theta_im2 = theta_im1.reshape(32,32)
        imshow(theta_im2, cm.coolwarm)
        figure = plt.gcf()
        plt.title(image_names[i] + '- Visualized Theta')
        plt.show()
        figure.savefig(os.getcwd() + "/results/Part 8 -" + image_names[i] +'_theta.png')
    return None    


if __name__ == "__main__":
    # Part 1
#    download_uncropped_images(os.getcwd() + "\\faces_subset.txt")
 
# #    Part 2 
#    reader = csv.reader(open(os.getcwd() + '\\Namelist\crop coordinates.csv', 'r'))
#    crop_coordinates = {}
#    for row in reader:
#       coordinate,image_name = row
#       crop_coordinates[image_name] = coordinate
#    image_list_uncroped = get_all_images(os.getcwd() + "\uncropped\*")
##    update_images(image_list_uncroped,crop_coordinates)
#
    uncropped_path = os.getcwd() + "\uncropped\*"
    cropped_path = os.getcwd() + "\cropped\*"
    act = list(set([a.split("\t")[0] for a in open(os.getcwd() + "\\faces_subset.txt").readlines()]))
    image_list = get_all_images(uncropped_path)
#    actor_name_dataframe_origin = pd.DataFrame.from_csv('actor names.csv', index_col = None)
    name_dict = {}
    for i in act:
        last_name = (i.split(' ')[1]).lower()
        for j in glob.glob(os.getcwd() + "\cropped\\" + last_name + "*"):
            image_name = j.split("\\")[-1]
            name_dict[image_name] = i
                   
    actor_name_dataframe_origin = pd.DataFrame.from_dict(name_dict,orient='index')
    actor_name_dataframe_origin['image_name'] = actor_name_dataframe_origin.index
    actor_name_dataframe_origin = actor_name_dataframe_origin.rename(columns = {0: 'actor_names'})


    actor_dataframe = clear_non_readable_files(actor_name_dataframe_origin, image_list)[1]
    image_name_list = clear_non_readable_files(actor_name_dataframe_origin, image_list)[0]
    organized_data = organize_all_data(act, actor_dataframe, image_name_list, cropped_path)
    subset_data = organize_subset_data(act, actor_dataframe, image_name_list, cropped_path)

#   Part 3
    Bill_Data = add_Bill_label(organized_data['Bill Hader'])
    Steve_Data = add_Steve_label(organized_data['Steve Carell'])
    regression_sets = build_regression_sets_two_actors(Bill_Data, Steve_Data)
    initial_theata = np.array([0]*1025).reshape(1025,1)
#
    alpha = 0.00001
    theta_full = train_regression_model(regression_sets[0], f, df, alpha)[0]
    cost_function_value_train = train_regression_model(regression_sets[0], f, df, alpha)[1]
    
    two_per_act_train = two_per_actor_training_set(Bill_Data,Steve_Data)
    theta_two_act = train_regression_model(two_per_act_train, f, df, alpha)[0]

    trainging_set_accuracy = evaluate_performace(regression_sets[0],theta_full)[0]  
    validation_set_accuracy = evaluate_performace(regression_sets[1],theta_full)[0]
#    hypothesis_y_train = evaluate_performace(regression_sets[0],theta_full)[1]
    cost_function_value_validation = evaluate_performace(regression_sets[2],theta_full)[1]
    
    print "The cost function value on training set is " + str(cost_function_value_train)
    print "The Accuracy on the Training Set is: "+ str( trainging_set_accuracy *100) + "% \n"
    print "The cost function value on validation set is " + str(cost_function_value_validation)
    print "The Accuracy on the Validation Set is: "+ str( validation_set_accuracy *100) + "% \n"

#     Part 4
    visualize_theta(theta_full, "Part 4 - theta full.jpg") 
    visualize_theta(theta_two_act, "Part 4 - theta two images.jpg")                                              
#    
##     Part 5
#    alpha = 0.000001
#    gender_labeled_sets = add_gender_labels(subset_data)
#    model_stats = get_model_stats(train_regression_model, evaluate_performace, build_regression_sets_gender, f, df)
#    plot_model_stats(model_stats)
#    training_data_gen = build_regression_sets_gender(gender_labeled_sets, 1000)[0]
#    theta_gender = train_regression_model(training_data_gen, f, df, alpha)
#    non_act_test_data = build_regression_sets_gender(gender_labeled_sets, 1000)[2]
#    test_performance = evaluate_performace(non_act_test_data, theta_gender)[0]
#    print "The Accuracy on the Non-Act Test Set is: "+ str( test_performance *100) + "% \n"
##
##    
#    # Part 6 
##    x =  np.random.rand(5,6)
##    y = np.random.rand(4,6)
##    theta = np.random.rand(5,4)
#
#
#    one_hot_full_data = one_hot_encode_set(subset_data)
##    
#    regression_set_multi = separate_training_set_multiclass('Fran Drescher', one_hot_full_data)
#    act_sub = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
#    training_df = build_regression_sets_multi_class(act_sub, separate_training_set_multiclass)[0]
#    [training_x_multi, training_y_multi] = get_regression_parameters_multiclass(training_df)
#    
#    gradient_trial_x = training_x_multi[0:5,0:6]
#    gradient_trial_y = training_y_multi[0:4,0:6]
#    np.random.seed(1)
#    gradient_trial_theta = np.random.rand(5,4)
#    df = df_multi_class(gradient_trial_x, gradient_trial_y, gradient_trial_theta)
#    df_FD = finite_difference(gradient_trial_x, gradient_trial_y, gradient_trial_theta)
#    print 'The derivative of cost funtion computated by df_multi_class is \n', df
#    print 'The derivative of cost funtion computated by finite difference is \n',  df_FD
#    error = df - df_FD
#    print 'The difference between two methods is \n',error
##    
#    
#    # Part 7
#
#    alpha = 0.000001
#    theta_multi = train_regression_model_multi(training_df, f_multi_class, df_multi_class, alpha)
#    test_df = build_regression_sets_multi_class(act_sub, separate_training_set_multiclass)[1]
#    validation_df = build_regression_sets_multi_class(act_sub, separate_training_set_multiclass)[2]
#    validation_set = pd.concat([test_df, validation_df])
#    
#    multi_class_performance_training =  evaluate_regression_model_multi(training_df,theta_multi)
#    multi_class_performance_validation =  evaluate_regression_model_multi(validation_set,theta_multi)
#    
#    print "The Accuracy on the Training Set is: "+ str( multi_class_performance_training *100) + "% \n"
#    print "The Accuracy on the Validation Set is: "+ str( multi_class_performance_validation *100) + "% \n"
#    
#    # Part 8
#    visualize_theta_multi(theta_multi, training_df)
#### 