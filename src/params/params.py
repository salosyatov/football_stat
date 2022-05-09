from dataclasses import dataclass, field
from typing import Optional, List


@dataclass()
class PreprocessingParams:
    input_data_path: str = field(default="./data/raw")
    output_cache_path: str = field(default="./data/interim")
    cache_file_path: str = field(default="./data/interim/cache.npz")
    processed_images_folder: str = field(default="./data/processed")
    height: int = field(default=80)
    width: int = field(default=32)
    save_processed_images: bool = field(default=False)


@dataclass()
class DownloadParams:
    paths: List[str]
    output_folder: str
    #s3_bucket: str = field(default="for-dvc")


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)


@dataclass()
class TrainingParams:
    model_net_type: str = field(default="TripletLoss")
    embedding_size: str = field(default=64)
    model_clf_type: str = field(default="XGBClassifier")
    seed: int = field(default=0)
    random_state: int = field(default=255)
    epochs: int = field(default=1)
    batch_size: int = field(default=32)


@dataclass()
class TestParams:
    model_net_type: str = field(default="TripletLoss")
    embedding_size: str = field(default=64)
    model_clf_type: str = field(default="XGBClassifier")
    seed: int = field(default=0)
    random_state: int = field(default=255)


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_net_model_path: str
    output_clf_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    preprocessing_params: PreprocessingParams
    train_params: TrainingParams
    test_params: TestParams
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = True
    mlflow_uri: str = "http://localhost/"
    mlflow_experiment: str = "inference_demo"
    use_cache: bool = True
