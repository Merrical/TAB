from models.TAB import build_TAB


def build_model(args):
    """
    return models
    """
    if args.net_arch == "TAB":
        model = build_TAB(args)
    return model

