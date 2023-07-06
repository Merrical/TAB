# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build_TAB_model


def build_TAB(args):
    return build_TAB_model(args)
