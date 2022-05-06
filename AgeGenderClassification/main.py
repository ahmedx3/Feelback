import pickle as pickle
# import FeaturesExtraction
import AgeGenderClassification.FeaturesExtraction as FeaturesExtraction
# import Preprocessing
import AgeGenderClassification.Preprocessing as Preprocessing
from collections import defaultdict

import AgeGenderClassification.Utils  as Utils

class GenderAgeClassification:
    def __init__(self, model_path_age, model_path_gender):
        self.modelAge = pickle.load(open(model_path_age, 'rb'))
        self.modelGender = pickle.load(open(model_path_gender, 'rb'))
        self.previous_gender_values = defaultdict(list)

    def split_prob(self, prediction):
        split_pred = prediction.split('-')
        return split_pred[0], split_pred[1]
        
    def get_max_prob_gender(self, id, prediction, method="top_votes"):
        # Split probability and prediction
        # initialize the max to be the current
        max_pred = self.split_prob(prediction)

        if method == "single_max":
            # Compare with the previous values
            for pred in self.previous_gender_values[id]:
                pred = self.split_prob(pred)
                if float(max_pred[1]) < float(pred[1]):
                    max_pred = pred

            # Add previous value to array
            self.previous_gender_values[id].append(prediction)
        
        elif method == "top_votes":
            # Max number of frames to take average
            max_kept = 5

            # If we did not reach max_kept add it to the list
            if len(self.previous_gender_values[id]) < max_kept:
                # Add previous value to array
                self.previous_gender_values[id].append(prediction)
                
            # Else we loop and replace the least probability
            else:
                least_prob_index = None
                least_prob_value = 1
                # Compute least probability in the list
                for index, pred in enumerate(self.previous_gender_values[id]):
                    pred = self.split_prob(pred)
                    if float(pred[1]) < least_prob_value:
                        least_prob_index = index
                        least_prob_value = float(pred[1])
                
                # Check if the least probability is less than the current and replace
                curr_pred = self.split_prob(prediction)
                if float(curr_pred[1]) > least_prob_value:
                    self.previous_gender_values[id][least_prob_index] = prediction
                    
            
            # get the average over the max_kept
            male_votes, female_votes = 0, 0
            male_prob, female_prob = 0, 0

            loop_number = min(len(self.previous_gender_values[id]), max_kept)
            for index in range(loop_number):
                pred = self.split_prob(self.previous_gender_values[id][index])
                if pred[0] == 'male':
                    male_votes += 1
                    male_prob += float(pred[1])
                else:
                    female_votes += 1
                    female_prob += float(pred[1])
            
            # compute the max prediction value from the votes
            max_class = 'male' if male_votes > female_votes else 'female'
            max_prob = round(male_prob / male_votes, 2) if male_votes > female_votes else round(female_prob / female_votes, 2)
            max_pred = (max_class, max_prob)
            
        return max_pred

    def getGender(self, frame, facesLocations, ids):
        faces = []
        for x1,y1,x2,y2 in facesLocations:
            faces.append(frame[y1-70:y2+40,x1-10:x2+10])

        predictedGender = []
        for index, face in enumerate(faces):
            # Preprocess and extract features
            img = Preprocessing.preprocess_image(face)
            img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
            
            # Predict the probability
            predicted_prob = self.modelGender.predict_proba([img_features])[0]
            
            # Compute the prediction based on the criteria chosen
            prediction = str(self.modelGender.predict([img_features])[0]) + '-' + str(round(max(predicted_prob), 2))
            prediction = self.get_max_prob_gender(ids[index], prediction, method="top_votes")
            
            # Add the prediction to the list of predioctions
            prediction = str(prediction[0]) + ' ' + str(prediction[1])
            predictedGender.append(prediction)

        return predictedGender
    
    def getAge(self, faces):
        predictedAge = []
        for face in faces:
            img = Preprocessing.preprocess_image(face)
            img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
            predictedAge.append(self.modelAge.predict([img_features])[0])

        return predictedAge

            
