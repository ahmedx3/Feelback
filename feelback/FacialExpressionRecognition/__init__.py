# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])


import pickle as pickle
from . import FeaturesExtraction


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


