# logger_config.py

import logging
import logging.config


def setup_logging():
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(levelname)s:%(name)s: %(message)s',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'level': 'INFO',
                },
            },
            'loggers': {
                # Enable INFO output for this specific script/module
                'physana.routines.run_histmaker_json': {
                    'handlers': ['console'],
                    'level': 'INFO',
                    'propagate': False,
                },
                # Keep histmaker quiet unless elevated
                'physana.algorithm.histmaker': {
                    'handlers': ['console'],
                    'level': 'ERROR',
                    'propagate': False,
                },
                'distributed': {
                    'handlers': ['console'],
                    'level': 'ERROR',
                    'propagate': False,
                },
                'distributed.core': {'level': 'ERROR'},
                'distributed.worker': {'level': 'ERROR'},
                'distributed.nanny': {'level': 'ERROR'},
            },
            'root': {
                'handlers': ['console'],
                'level': 'WARNING',
            },
        }
    )

    # Remove any old handlers lingering on the logger
    logger = logging.getLogger('physana.routines.run_histmaker_json')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add a fresh console handler with INFO level
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def set_verbose_histmaker():
    logger = logging.getLogger('physana.algorithm.histmaker')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)
