import pickle as pickle
import FacialExpressionRecognition.FeaturesExtraction as FeaturesExtraction

class EmotionExtraction:
    def __init__(self, model_path):
        self.featureExtractor = FeaturesExtraction.FeatureExtractor(True)
        self.model = pickle.load(open(model_path, 'rb'))

    def getEmotion(self, faces):
        predictedEmotion = []
        for face in faces:
            img_features = self.featureExtractor.extract_features(face, feature="LANDMARKS")
            predictedEmotion.append(self.model.predict([img_features])[0])

        return predictedEmotion
    
            
