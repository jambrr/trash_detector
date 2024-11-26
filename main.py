from src.data.preprocessor import TacoPreprocessor, TrashnetPreprocessor
from src.models.trainers import Yolov11Trainer
import logging
logger = logging.getLogger(__name__)

# preprocessor = TacoPreprocessor("taco", "datasets/TACO", "datasets/TACO/data/annotations.json")
preprocessor = TrashnetPreprocessor("trashnet", "datasets/trashnet", "datasets/trashnet/train/_annotations.coco.json",  "datasets/trashnet/valid/_annotations.coco.json")

preprocessor.preprocess(preprocessor.train_df)

print(len(preprocessor.augmented_df))
