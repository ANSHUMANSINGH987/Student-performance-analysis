import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
	def __init__(self):
		self.data_ingestion = DataIngestion()
		self.data_transformation = DataTransformation()
		self.model_trainer = ModelTrainer()

	def start_data_ingestion(self):
		logging.info("Starting data ingestion stage")
		return self.data_ingestion.initiate_data_ingestion()

	def start_data_transformation(self, train_path, test_path):
		logging.info("Starting data transformation stage")
		return self.data_transformation.initiate_data_transformation(train_path, test_path)

	def start_model_training(self, train_array, test_array):
		logging.info("Starting model training stage")
		return self.model_trainer.initiate_model_trainer(train_array, test_array)

	def run_pipeline(self):
		try:
			train_path, test_path = self.start_data_ingestion()
			train_array, test_array, _ = self.start_data_transformation(train_path, test_path)
			return self.start_model_training(train_array, test_array)

		except Exception as e:
			raise CustomException(e, sys)


if __name__ == "__main__":
	pipeline = TrainPipeline()
	print(pipeline.run_pipeline())
