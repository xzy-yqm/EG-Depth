from .kitti_dataset import KITTIDataset
from .kitti_dataset_completion import KITTIDatasetCompletion
from .sceneflow_dataset import SceneFlowDatset
from .zjlab_dataset import ZjlabDataset
__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "kitti_completion": KITTIDatasetCompletion,
    "zjlab": ZjlabDataset
}
