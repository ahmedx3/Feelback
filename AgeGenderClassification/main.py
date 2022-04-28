import pickle as pickle
# import FeaturesExtraction
import AgeGenderClassification.FeaturesExtraction as FeaturesExtraction
# import Preprocessing
import AgeGenderClassification.Preprocessing as Preprocessing

import AgeGenderClassification.Utils  as Utils

class GenderAgeClassification:
    def __init__(self, model_path_age, model_path_gender):
        self.modelAge = pickle.load(open(model_path_age, 'rb'))
        self.modelGender = pickle.load(open(model_path_gender, 'rb'))

    def getGender(self, frame, facesLocations):
        faces = []
        for x1,y1,x2,y2 in facesLocations:
            faces.append(frame[y1-70:y2+40,x1-10:x2+10])

        predictedGender = []
        for face in faces:
            img = Preprocessing.preprocess_image(face)
            img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
            predicted_prob = self.modelGender.predict_proba([img_features])[0]
            prediction = str(self.modelGender.predict([img_features])[0]) + ' ' + str(round(max(predicted_prob), 2))
            predictedGender.append(prediction)

        return predictedGender
    
    def getAge(self, faces):
        predictedAge = []
        for face in faces:
            img = Preprocessing.preprocess_image(face)
            img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
            predictedAge.append(self.modelAge.predict([img_features])[0])

        return predictedAge

            
