from crystal_gnn.models.cgcnn import CGCNN
from crystal_gnn.models.schnet import SCHNET
from crystal_gnn.models.megnet import MEGNET
from crystal_gnn.experimental.alignn import ALIGNN
from crystal_gnn.experimental.nequip.nequip import NEQUIP
from crystal_gnn.models.cgcnn_global import CGCNNGlobal

_models = {
    "schnet": SCHNET,
    "cgcnn": CGCNN,
    "megnet": MEGNET,
    "alignn": ALIGNN,
    "nequip": NEQUIP,
    "cgcnn_global": CGCNNGlobal,
}
