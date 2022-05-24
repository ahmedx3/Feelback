# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import pickle as pickle
from . import FeaturesExtraction
from . import Preprocessing
from . import Utils
from collections import defaultdict
import numpy as np


class AgeGenderClassification:
    def __init__(self, model_path_age, model_path_gender):
        self.modelAge = pickle.load(open(model_path_age, 'rb'))
        self.modelGender = pickle.load(open(model_path_gender, 'rb'))
        self.previous_gender_values = defaultdict(list)
        self.previos_age_values = defaultdict(list)

    def split_prob(self, prediction):
        split_pred = prediction.split('-')
        return split_pred[0], split_pred[1]

    def get_max_prob_gender(self, id, prediction, method="same_frame"):
        # Split probability and prediction
        # initialize the max to be the current
        max_pred = self.split_prob(prediction)

        if method == "same_frame":
            # return the same value that we got
            return max_pred

        elif method == "single_max":
            # Compare with the previous value
            if len(self.previous_gender_values[id]) == 1:
                current_max = self.split_prob(self.previous_gender_values[id][0])
                if float(max_pred[1]) > float(current_max[1]):
                    self.previous_gender_values[id][0] = prediction
                else:
                    max_pred = self.split_prob(self.previous_gender_values[id][0])
            else:
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

            # TODO: REMOVE AFTER TESTING
            # if id == 1:
            #     print(self.previous_gender_values[id])

        return max_pred

    def getGender(self, frame, facesLocations, ids):
        faces = []
        # count = 0
        for x1,y1,x2,y2 in facesLocations:
            # TODO: REMOVE AFTER TESTING
            # if count == 1:
            #     choicesx = np.arange(10, 100, 10)
            #     choicesy = [10,20]
            #     r1 = np.random.choice(choicesy)
            #     r2 = np.random.choice(choicesy)
            #     r3 = np.random.choice(choicesx)
            #     r4 = np.random.choice(choicesx)
            #     faces.append(frame[y1-r1:y2+r2,x1-r3:x2+r4])
            # else:
            #     faces.append(frame[y1-70:y2+40,x1-10:x2+10])

            # count += 1
            faces.append(frame[y1-70:y2+40,x1-10:x2+10])


        predictedGender = []
        for index, face in enumerate(faces):
            # If exception is raised return error label
            try:
                # Preprocess and extract features
                img = Preprocessing.preprocess_image(face)
                img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
                # Predict the probability
                predicted_prob = self.modelGender.predict_proba([img_features])[0]

                # Compute the prediction based on the criteria chosen
                prediction = str(self.modelGender.predict([img_features])[0]) + '-' + str(round(max(predicted_prob), 2))
                prediction = self.get_max_prob_gender(ids[index], prediction, method="single_max")

                # Add the prediction to the list of predioctions
                prediction = str(prediction[0]) + ' ' + str(prediction[1])
                predictedGender.append(prediction)

            except:
                predictedGender.append("ERROR")

        return predictedGender
    

    def get_age_helper(self, id, prediction, method="same_frame"):

        # Cast prediction to int
        prediction = round(prediction)
        
        if method == "same_frame":
            # return the same value that we got
            return prediction

        elif method == "average":
            # Add the age to the list corresponding to the ID
            self.previos_age_values[id].append(prediction)

            # Calculate average
            average_age = np.sum(np.array(self.previos_age_values[id])) / len(self.previos_age_values[id])

            # Cast prediction to int
            average_age = round(average_age)
        
            return average_age


    def getAge(self, frame, facesLocations, ids):

        faces = []
        for x1,y1,x2,y2 in facesLocations:
            faces.append(frame[y1-70:y2+40,x1-10:x2+10])

        predictedAge = []
        for index, face in enumerate(faces):
            # If exception is raised return error label
            try:
                # Preprocess and extract features
                img = Preprocessing.preprocess_image(face)
                img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
                # Predict the probability
                prediction = self.modelAge.predict([img_features])[0]
                
                # Compute the prediction based on the criteria chosen
                prediction = self.get_age_helper(ids[index], prediction, method="average")
                
                # Add the prediction to the list of predioctions
                predictedAge.append(prediction)

            except:
                predictedAge.append("ERROR")

        return predictedAge

    # TODO: Implement
    def getFinalGenders(self, ids: np.ndarray) -> np.ndarray:
        return np.array(["Male"] * len(ids))

    # TODO: Implement
    def getFinalAges(self, ids: np.ndarray) -> np.ndarray:
        return np.array([25] * len(ids))
            
