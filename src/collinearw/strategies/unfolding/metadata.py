import logging

log = logging.getLogger(__name__)


def has(configMgr, key, name):
    """
    Returns False if metadata check is fine (no conflicting entries).
    Returns True if metadata check failed (conflicting entries).
    """
    configMgr.meta_data.setdefault(
        "unfold", {"regions": {}, "observables": {}, "processes": {}}
    )
    if name in configMgr.meta_data['unfold'][key]:
        log.warning(
            f"Seems {name} is already registered as one of the {key} in the configMgr cache."
        )
        return True
    return False


def save(configMgr, key, name, value):
    """
    Tries to save the configuration onto the configMgr.
    """
    if has(configMgr, key, name):
        log.warning(
            f"{name} was not saved as one of the {key} in the configMgr as it was already defined."
        )
        return False

    configMgr.meta_data["unfold"][key].setdefault(name, {}).update(value)
    return True


def processes(configMgr):
    return configMgr.meta_data['unfold']['processes']


def regions(configMgr):
    return configMgr.meta_data['unfold']['regions']


def observables(configMgr):
    return configMgr.meta_data['unfold']['observables']
