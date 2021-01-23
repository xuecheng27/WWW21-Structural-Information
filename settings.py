
LOGGING_SETTINGS = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'detail': {
            'format': "[%(asctime)s] %(levelname)s [%(process)d] %(message)s",
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'simple': {
            'format': '%(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024 * 1024 * 30,
            'backupCount': 20,
            'delay': False,
            'filename': 'logs/test.log',
            'formatter': 'simple',
            'encoding': 'utf8'
        }
    },
    'loggers': {
        'normal': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG'
        },
        'console': {
            'handlers': ['console'],
            'level': 'DEBUG'
        }
    }
}