from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, AugTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "base_aug": AugTrainer
}
