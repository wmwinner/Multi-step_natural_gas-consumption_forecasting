from dataclasses import dataclass, field, asdict
from config.BaseParams import BaseParams


@dataclass
class HATL_Seq2SeqParams(BaseParams):
    # =============================================
    # 🤖 模型结构参数
    # =============================================
    model_name: str = field(default="HATL_Seq2Seq", metadata={"help": "模型名称"})
    lr: float = field(default=0.0001, metadata={"help": "学习率"})
    e_layers: int = field(default=1, metadata={"help": "Encoder编码层层数"})
    d_layers: int = field(default=1, metadata={"help": "Decoder解码层层数"})

    d_model: int = field(default=256, metadata={"help": "编码后的长度"})
    factor: int = field(default=3, metadata={"help": "attn factor"})
    d_ff: int = field(default=128, metadata={"help": "dimension of fcn"})

    top_k: int = field(default=5, metadata={"help": "for TimesBlock"})
    n_heads: int = field(default=4, metadata={"help": "num of heads"})

    dropout: float = field(default=0.1, metadata={"help": "Dropout率"})

    kernel_size: (int, int) = field(
        default=(3, 3), metadata={"help": "隐藏层神经元数量"}
    )
